"""
文件: train_sac.py
说明: SAC 训练脚本（时间/步数/两阶段等参数全部从 config.py/GUI 读取）
依赖:
  - stable_baselines3
  - 项目内: power_grid_env.py, config.py, training_visualizer.py
"""

import os
import traceback
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from power_grid_env import PowerGridEnv
from training_visualizer import CostCurveCallback
from config import PATHS, CORE_PARAMS, TRAINING_CONFIG, load_gui_settings


def make_env(use_two_stage: bool):
    """
    构造环境：
    - 所有“时间相关”的参数（start_hour/end_hour/step_minutes）来自 GUI 设置或 CORE_PARAMS
    - 不在脚本内写死任何时间参数
    """
    gui_params = load_gui_settings()

    params = {
        "grid_model": gui_params.get("grid_model", CORE_PARAMS.get("grid_model", "ieee33")),
        "solver": gui_params.get("solver", CORE_PARAMS.get("solver", "gurobi")),
        "start_hour": gui_params.get("start_hour", CORE_PARAMS.get("start_hour", 1)),
        "end_hour": gui_params.get("end_hour", CORE_PARAMS.get("end_hour", 24)),
        "step_minutes": gui_params.get("step_minutes", CORE_PARAMS.get("step_minutes", 60)),
        "distributed_energy": {
            "pv":   gui_params.get("use_pv",   CORE_PARAMS.get("distributed_energy", {}).get("pv", True)),
            "wind": gui_params.get("use_wind", CORE_PARAMS.get("distributed_energy", {}).get("wind", True)),
            "ess":  gui_params.get("use_ess",  CORE_PARAMS.get("distributed_energy", {}).get("ess", True)),
        },
        "sop_nodes_active": gui_params.get("use_sop", CORE_PARAMS.get("sop_nodes_active", True)),
        "nop_nodes_active": gui_params.get("use_nop", CORE_PARAMS.get("nop_nodes_active", True)),
        "slack_bus": CORE_PARAMS.get("slack_bus", "b1"),
        "base_power": CORE_PARAMS.get("base_power", 1.0),
    }
    return PowerGridEnv(gui_params=params, use_two_stage_flow=use_two_stage)


def main():
    # ============== 路径与目录 ==============
    project_root = os.path.dirname(os.path.abspath(__file__))
    logs_root = PATHS.get("logs_dir", os.path.join(project_root, "logs"))
    models_root = os.path.join(project_root, "models")
    tb_root = os.path.join(project_root, "tensorboard_logs")
    os.makedirs(logs_root, exist_ok=True)
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(tb_root, exist_ok=True)

    # ============== 统一读取训练配置 ==============
    TWO_STAGE_TRAIN = bool(TRAINING_CONFIG.get("two_stage_training", True))
    TOTAL_TIMESTEPS = int(TRAINING_CONFIG.get("total_timesteps", 300_000))
    EVAL_FREQ = int(TRAINING_CONFIG.get("eval_freq", 5_000))
    CHECKPOINT_FREQ = int(TRAINING_CONFIG.get("checkpoint_freq", 10_000))
    COST_CURVE_EVAL_FREQ = int(TRAINING_CONFIG.get("cost_curve_eval_freq", 500))
    RANDOM_SEED = int(TRAINING_CONFIG.get("seed", 0))

    env_total = os.getenv("TRAINING_TOTAL_TIMESTEPS")
    if env_total is not None:
        try:
            TOTAL_TIMESTEPS = int(env_total)
        except ValueError:
            pass

    env_seed = os.getenv("TRAINING_SEED")
    if env_seed is not None:
        try:
            RANDOM_SEED = int(env_seed)
        except ValueError:
            pass


    # 可选：SAC 专属超参也从 TRAINING_CONFIG 读取（没有就使用默认）
    SAC_PARAMS = TRAINING_CONFIG.get("sac_params", {})
    learning_rate = SAC_PARAMS.get("learning_rate", 3e-4)
    buffer_size = SAC_PARAMS.get("buffer_size", 1_000_000)
    batch_size = SAC_PARAMS.get("batch_size", 256)
    tau = SAC_PARAMS.get("tau", 0.005)
    gamma = SAC_PARAMS.get("gamma", 0.99)
    train_freq = SAC_PARAMS.get("train_freq", 1)
    gradient_steps = SAC_PARAMS.get("gradient_steps", 1)

    algo_tag = "sac"
    LOG_DIR = os.path.join(logs_root, f"{algo_tag}_{'two_stage' if TWO_STAGE_TRAIN else 'one_stage'}")
    BEST_MODEL_DIR = os.path.join(models_root, f"best_{algo_tag}_{'2stage' if TWO_STAGE_TRAIN else '1stage'}")
    FINAL_MODEL_PATH = os.path.join(models_root, f"{algo_tag}_{'2stage' if TWO_STAGE_TRAIN else '1stage'}_final.zip")
    TENSORBOARD_LOG_DIR = os.path.join(tb_root, f"{algo_tag}_{'2stage' if TWO_STAGE_TRAIN else '1stage'}")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    print("=" * 80)
    print(f"[SAC] 日志目录: {LOG_DIR}")
    print(f"[SAC] 最优模型目录: {BEST_MODEL_DIR}")
    print(f"[SAC] TensorBoard: {TENSORBOARD_LOG_DIR}")
    print(f"[SAC] 训练步数: {TOTAL_TIMESTEPS:,}")
    print(f"[SAC] 两阶段训练: {TWO_STAGE_TRAIN}")
    print("=" * 80)

    # ============== 构造训练/评估环境 ==============
    try:
        # 评估环境固定两阶段，保证公平
        eval_env_raw = make_env(use_two_stage=True)
        eval_env = Monitor(eval_env_raw)

        # 训练环境按配置选择两阶段或单阶段（训练期可先关两阶段提速）
        train_env_raw = make_env(use_two_stage=TWO_STAGE_TRAIN)
        train_env = Monitor(train_env_raw)
        train_env = DummyVecEnv([lambda: train_env])

        # 统一随机种子
        np.random.seed(RANDOM_SEED)
        try:
            train_env.seed(RANDOM_SEED)
        except Exception:
            pass
        try:
            eval_env.seed(RANDOM_SEED)
        except Exception:
            pass

    except Exception as e:
        print(f"[SAC] 创建环境失败: {e}")
        traceback.print_exc()
        return

    # ============== 构建 SAC 模型 ==============
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        verbose=1,
        seed=RANDOM_SEED,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
    )

    # ============== 回调：评估/ckpt/成本曲线 ==============
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=BEST_MODEL_DIR,
        name_prefix="ckpt"
    )
    agent_name_for_plot = f"SAC_{'Two_Stage' if TWO_STAGE_TRAIN else 'Single_Stage'}"

    cost_curve_cb = CostCurveCallback(
        eval_env=eval_env_raw,
        agent_name=agent_name_for_plot,  # <--- (1) 添加缺失的 agent_name
        save_path=BEST_MODEL_DIR,  # <--- (2) 修正参数名 log_dir -> save_path
        eval_freq=COST_CURVE_EVAL_FREQ
    )

    # ============== 启动训练 ==============
    try:
        tb_run_name = f"seed_{RANDOM_SEED}"
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_cb, checkpoint_cb, cost_curve_cb],
            tb_log_name=tb_run_name,
            progress_bar=True,
        )

    except KeyboardInterrupt:
        print("\n[SAC] 收到中断信号，保存当前模型...")
    except Exception as e:
        print(f"[SAC] 训练异常: {e}")
        traceback.print_exc()
    finally:
        # 保存最终模型、关闭环境
        try:
            cost_curve_cb.save_data()
        except Exception:
            pass
        print(f"[SAC] 保存最终模型到: {os.path.abspath(FINAL_MODEL_PATH)}")
        model.save(FINAL_MODEL_PATH)
        try:
            model.logger.close()
        except Exception:
            pass
        try:
            train_env.close()
            eval_env.close()
        except Exception:
            pass
        print("[SAC] 训练结束。")


if __name__ == "__main__":
    main()
