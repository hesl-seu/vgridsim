# -*- coding: utf-8 -*-
"""
training_all_parallel.py
并行启动 SAC、DDPG、TD3、PPO 的训练，适合“跑一晚上”。
- 为每个算法开一个子进程（调用现有 train_*.py）
- 限制每个进程的 CPU 线程数量
- 统一归档每个子进程的模型/图表/数据到 runs/<run_id>/<Algo>/artifacts
"""
import argparse
import datetime as dt
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "SAC":  "train_sac.py",
    "DDPG": "train_ddpg.py",
    "TD3":  "train_td3.py",
    "PPO":  "train_ppo.py",
}
DEFAULT_ORDER = ["SAC", "DDPG", "TD3", "PPO"]
DEFAULT_SEEDS = [0, 1, 2]

def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _pump_to_console_and_file(prefix: str, pipe, logfile: Path):
    """把子进程的输出同时写到控制台与日志文件"""
    with open(logfile, "wb") as lf:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            try:
                sys.stdout.buffer.write(b"[" + prefix.encode("utf-8") + b"] " + chunk)
                sys.stdout.flush()
            except Exception:
                pass
            lf.write(chunk)

def _collect_artifacts(work_dir: Path, artifacts_dir: Path):
    """把子进程工作目录下常见产物拉到 artifacts"""
    patterns = [
        "*.zip", "*.pth", "*.pt",
        "*.png", "*.jpg",
        "*.pkl", "*.npz", "*.npy", "*.csv", "*.json",
        "events.*",  # TensorBoard
    ]
    _ensure(artifacts_dir)
    for pat in patterns:
        for src in work_dir.rglob(pat):
            # 避免把 artifacts 自己再复制进来
            if artifacts_dir in src.parents:
                continue
            dst = artifacts_dir / src.name
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass

def launch_one(algo: str, run_dir: Path, per_threads: int, nice: str, total_steps: int, seed: int, device: str, tag: str):
    script = SCRIPTS[algo]
    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        raise FileNotFoundError(f"{script} 不存在，请确认路径。")

    work_dir = _ensure(run_dir / algo)
    logs_dir = _ensure(work_dir / "logs")
    artifacts_dir = _ensure(work_dir / "artifacts")
    logfile = logs_dir / "train.log"

    # 进程环境
    env = os.environ.copy()
    # 给子脚本的“软参数通道”，子脚本可选读取，不读取也不影响
    if total_steps > 0:
        env["TRAINING_TOTAL_TIMESTEPS"] = str(total_steps)
    if seed >= 0:
        env["TRAINING_SEED"] = str(seed)
    if device:
        env["TRAINING_DEVICE"] = device
    env["RUN_TAG"] = tag

    # 限制每进程 BLAS/OMP 线程，防止互抢
    for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[k] = str(per_threads)

    # 轻量 lower priority（*nix 有效；Windows 忽略）
    pre_cmd = []
    if nice in ("low", "idle") and os.name == "posix":
        level = {"low": "10", "idle": "19"}[nice]
        pre_cmd = ["nice", "-n", level]

    cmd = pre_cmd + [sys.executable, str(script_path)]

    print(f"[LAUNCH] {algo}: {' '.join(cmd)}")
    print(f"[DIR]    {work_dir}")

    # 启动子进程
    proc = subprocess.Popen(
        cmd,
        cwd=work_dir,              # 每个算法用独立工作目录
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0
    )
    t = threading.Thread(target=_pump_to_console_and_file, args=(algo, proc.stdout, logfile), daemon=True)
    t.start()
    return proc, work_dir, artifacts_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER, help="要同时训练的算法")
    ap.add_argument("--per-proc-threads", type=int, default=2, help="每个进程的BLAS/OMP线程数")
    ap.add_argument("--nice", choices=["normal", "low", "idle"], default="low", help="进程优先级（仅 *nix 有效）")
    ap.add_argument("--total-steps", type=int, default=0, help="过夜步数（通过环境变量传给子脚本）")
    ap.add_argument("--seed", type=int, default=-1, help="训练随机种子（通过环境变量传给子脚本）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[], help="多随机种子，例如: --seeds 0 1 2 3 4")
    ap.add_argument("--device", choices=["cpu", "cuda", ""], default="cpu", help="期望设备，传给子脚本参考")
    ap.add_argument("--tag", type=str, default="overnight", help="本次跑的标签")
    args = ap.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{args.tag}"
    base_run_dir = _ensure(PROJECT_ROOT / "runs" / run_id)
    print(f"[RUN] 运行ID: {run_id}")
    print(f"[RUN] 根目录: {base_run_dir}")

    # 1) 决定要用哪些种子
    if args.seeds:  # 优先使用命令行传入的 --seeds
        seed_list = args.seeds
    elif args.seed >= 0:  # 其次用单个 --seed
        seed_list = [args.seed]
    else:  # 都没给，就用 DEFAULT_SEEDS
        seed_list = DEFAULT_SEEDS

    procs = []
    try:
        for sd in seed_list:
            print("\n" + "=" * 80)
            print(f"[SEED] 开始训练 seed = {sd}")
            print("=" * 80)

            # 每个 seed 用一个子目录：runs/<run_id>/seed_<sd>/
            run_dir = _ensure(base_run_dir / f"seed_{sd}")

            procs.clear()
            for algo in args.algos:
                proc, work_dir, artifacts_dir = launch_one(
                    algo=algo,
                    run_dir=run_dir,
                    per_threads=args.per_proc_threads,
                    nice=args.nice,
                    total_steps=args.total_steps,
                    seed=sd,  # ★ 把当前 seed 传下去
                    device=args.device,
                    tag=f"{args.tag}_seed{sd}",  # ★ RUN_TAG 中带上 seed
                )
                procs.append((algo, proc, work_dir, artifacts_dir))

            # 等待该 seed 下的所有算法结束
            exit_codes = {}
            for algo, proc, work_dir, artifacts_dir in procs:
                code = proc.wait()
                exit_codes[algo] = code
                if code == 0:
                    print(f"[{algo}] 训练结束，退出码={code}，产物已收集至 {artifacts_dir}")
                else:
                    print(f"[{algo}] 训练失败，退出码={code}，请查看 {work_dir}/logs/train.log")

            failed = [a for a, c in exit_codes.items() if c != 0]
            if failed:
                print(f"[SUMMARY seed={sd}] 有失败的进程：", failed)
            else:
                print(f"[SUMMARY seed={sd}] 全部训练进程成功结束。")

    except KeyboardInterrupt:
        print("\n[ABORT] 收到 Ctrl+C，尝试终止所有子进程...")
        for _, proc, _, _ in procs:
            with contextlib.suppress(Exception):
                if proc.poll() is None:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGTERM)
                    else:
                        proc.terminate()
        sys.exit(130)


if __name__ == "__main__":
    import contextlib
    main()
