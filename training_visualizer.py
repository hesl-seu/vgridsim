# file: training_visualizer.py (最终完整版)

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from stable_baselines3.common.callbacks import BaseCallback
import re
# 导入您项目中的相关模块
from grid_model import create_grid
from evaluate_agents import evaluate_baseline

# 设置matplotlib以正确显示中文和负号
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# --- Part 1: 用于在训练时收集数据的回调类 ---

class CostCurveCallback(BaseCallback):
    """
    (数据收集模式)
    一个自定义的回调函数，其主要职责是：
    1. 在训练开始时，计算并分别保存Baseline的“纯运营成本”和“总目标值”。
    2. 在训练过程中，周期性地评估当前RL模型，并记录其总成本和总奖励。
    3. 在训练结束或被中断时，将收集到的所有数据保存到一个唯一的文件中。
    """

    def __init__(self, eval_env, agent_name: str, eval_freq: int = 5000, save_path: str = "./logs/", seed: int = 0):
        super(CostCurveCallback, self).__init__(verbose=1)
        self.eval_env = eval_env
        self.agent_name = agent_name
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.seed = seed

        # 用于存储数据的列表
        self.training_steps = []
        self.rl_costs = []
        self.rl_rewards = []  # 缩放后的奖励
        self.rl_rewards_unscaled = []  # 未缩放奖励

        # 定义两个独立的Baseline基准文件路径
        self.baseline_cost_file = os.path.join(self.save_path, "baseline_cost.pkl")
        self.baseline_objective_file = os.path.join(self.save_path, "baseline_objective.pkl")

        # 定义RL智能体的数据文件路径
        self.rl_data_file = os.path.join(self.save_path, f"{self.agent_name}_data.pkl")

    def _on_training_start(self) -> None:
        """在训练开始时，计算并分别保存Baseline的纯成本和总目标值"""
        os.makedirs(self.save_path, exist_ok=True)

        # 只要任一基准文件不存在，就重新进行一次完整的Baseline评估
        if not os.path.exists(self.baseline_cost_file) or not os.path.exists(self.baseline_objective_file):
            print("--- CostCurveCallback: 未找到Baseline基准文件，正在重新计算... ---")

            # 创建一个全新的、干净的电网实例用于Baseline评估
            grid_for_baseline = create_grid(
                model=self.eval_env.params['grid_model'],
                gui_params=self.eval_env.params
            )

            # 使用与RL环境相同的参数生成充电场景
            stations_list_for_baseline = self.eval_env.stations_list
            for i, station in enumerate(stations_list_for_baseline):
                num_evs = self.eval_env.stations_info[i]['Num_EVs_to_Generate']
                station.generate_daily_scenarios(num_evs_to_generate=num_evs)

            # 调用评估函数
            metrics, _ = evaluate_baseline(
                gui_params=self.eval_env.params,
                seed=self.seed,
                stations_list=stations_list_for_baseline,
                grid=grid_for_baseline,
                use_two_stage=self.eval_env.use_two_stage_flow
            )

            # 提取总目标值 (包含惩罚项)
            baseline_objective_value = metrics.get("总目标值")
            if baseline_objective_value is not None:
                with open(self.baseline_objective_file, 'wb') as f:
                    pickle.dump(baseline_objective_value, f)
                print(f"--- Baseline【总目标值】已保存: {baseline_objective_value:.2f} ---")

            # 计算纯运营成本 (不含惩罚项)
            baseline_operational_cost = (
                    metrics.get("购电成本", 0) +
                    metrics.get("发电成本", 0) +
                    metrics.get("SOP损耗成本", 0) +
                    metrics.get("ESS放电成本", 0)
            )
            with open(self.baseline_cost_file, 'wb') as f:
                pickle.dump(baseline_operational_cost, f)
            print(f"--- Baseline【纯运营成本】已保存: {baseline_operational_cost:.2f} ---")

        else:
            print("--- CostCurveCallback: 已找到存在的Baseline基准文件。 ---")

    def _on_step(self) -> bool:
        """周期性地评估并记录RL模型的成本和奖励"""
        if self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            print(f"\n--- CostCurveCallback ({self.agent_name}): 正在评估训练步数 {self.num_timesteps}... ---")

            obs, _ = self.eval_env.reset(seed=self.seed)
            terminated = False
            truncated = False

            episode_total_cost = 0
            episode_total_reward = 0
            episode_total_reward_unscaled = 0

            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

                step_cost = (
                        info.get('grid_purchase_cost', 0)
                        + info.get('generation_cost', 0)
                        + info.get('sop_loss_cost', 0)
                        + info.get('ess_discharge_cost', 0)
                )
                episode_total_cost += step_cost

                # 缩放后的奖励（环境返回）
                episode_total_reward += reward

                # 未缩放奖励，从 info 里取（step() 里刚加的）
                episode_total_reward_unscaled += info.get('reward_unscaled', 0.0)

            self.training_steps.append(self.num_timesteps)
            self.rl_costs.append(episode_total_cost)
            self.rl_rewards.append(episode_total_reward)
            self.rl_rewards_unscaled.append(episode_total_reward_unscaled)

            print(f"--- 评估完成。当前成本: {episode_total_cost:.2f}元 | 当前奖励: {episode_total_reward:.2f} ---")
        return True

    def save_data(self):
        """一个可被外部调用的、专门用于保存数据的方法"""
        if not self.training_steps:
            print(f"--- CostCurveCallback ({self.agent_name}): 警告 - 没有收集到任何评估数据，不创建数据文件。---")
            print(f"--- (这通常是因为训练步数太少，未达到评估频率 eval_freq={self.eval_freq}) ---")
            return

        print(f"--- CostCurveCallback ({self.agent_name}): 正在保存成本和奖励数据... ---")
        data_to_save = {
            'steps': self.training_steps,
            'costs': self.rl_costs,
            'rewards': self.rl_rewards,  # 缩放后
            'rewards_unscaled': self.rl_rewards_unscaled  # 未缩放
        }
        with open(self.rl_data_file, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"--- {self.agent_name} 的数据已成功保存至: {os.path.abspath(self.rl_data_file)} ---")

    def _on_training_end(self) -> None:
        """当训练正常结束时，自动调用保存数据的方法"""
        self.save_data()


# --- Part 2: 用于在所有训练结束后，独立运行以生成最终对比图的主程序 ---

# def plot_curves():
#     """
#     (绘图模式)
#     加载所有保存的成本/奖励数据，并绘制两张最终的对比图。
#     """
#     print("\n" + "=" * 50 + "\n--- 启动最终训练曲线绘图程序 ---")
#
#     import glob, os
#     from pathlib import Path
#     from config import PATHS
#
#     # ========== 新增：统一的“最新文件查找器” ==========
#     # 在 PATHS["models_dir"] 与 项目根目录/runs 下递归查找文件，按 mtime 取最新
#     def _find_latest(filename: str) -> str:
#         bases = []
#         # 1) models 目录（原有逻辑）
#         if "models_dir" in PATHS and os.path.isdir(PATHS["models_dir"]):
#             bases.append(PATHS["models_dir"])
#         # 2) runs 目录（过夜脚本产物归档处）
#         runs_dir = str((Path(__file__).resolve().parent / "runs"))
#         if os.path.isdir(runs_dir):
#             bases.append(runs_dir)
#
#         cands = []
#         for base in bases:
#             # 递归搜所有同名文件（同时覆盖 best_*two_stage* 与 artifacts 目录两种情形）
#             cands.extend(glob.glob(os.path.join(base, "**", filename), recursive=True))
#         if not cands:
#             return ""
#         # 选最近修改的一个
#         return max(cands, key=os.path.getmtime)
#
#     # ========== 新增：按智能体名自动找对应数据文件 ==========
#     def _find_agent_pkl(agent_name: str) -> str:
#         # 约定文件名：<AgentName>_data.pkl，如 TD3_Two_Stage_data.pkl
#         return _find_latest(f"{agent_name}_data.pkl")
#
#     # 要绘制的智能体
#     agent_names = ["SAC_Two_Stage", "DDPG_Two_Stage", "TD3_Two_Stage", "PPO_Two_Stage"]
#     agents_to_plot = {name: _find_agent_pkl(name) for name in agent_names}
#
#     # baseline 文件（谁先更新都行，取最新时间）
#     baseline_cost_file = _find_latest("baseline_cost.pkl")
#     baseline_objective_file = _find_latest("baseline_objective.pkl")
#
#     # 1. 加载所有数据
#     try:
#         with open(baseline_cost_file, 'rb') as f:
#             baseline_cost = pickle.load(f)
#         print(f"成功加载Baseline纯运营成本: {baseline_cost:.2f}  @ {baseline_cost_file}")
#     except Exception:
#         print(f"警告: 未找到Baseline成本文件: {baseline_cost_file}")
#         baseline_cost = None
#
#     try:
#         with open(baseline_objective_file, 'rb') as f:
#             baseline_objective = pickle.load(f)
#         print(f"成功加载Baseline总目标值: {baseline_objective:.2f}  @ {baseline_objective_file}")
#     except Exception:
#         print(f"警告: 未找到Baseline目标值文件: {baseline_objective_file}")
#         baseline_objective = None
#
#     agent_data = {}
#     for agent_name, data_file in agents_to_plot.items():
#         if not data_file:
#             print(f"警告: 未找到 {agent_name} 的数据文件，将跳过该模型。")
#             continue
#         try:
#             with open(data_file, 'rb') as f:
#                 data = pickle.load(f)
#                 # 兼容旧版本：没有 unscaled 时，就用 scaled 顶上
#                 if 'rewards_unscaled' not in data and 'rewards' in data:
#                     data['rewards_unscaled'] = data['rewards']
#                 agent_data[agent_name] = data
#             print(f"成功加载 {agent_name} 的数据: {data_file}")
#         except Exception as e:
#             print(f"警告: 打开 {agent_name} 的数据文件失败({data_file})：{e}")
#
#     # 如果完全没加载到任何agent数据，给出可能原因提示（eval_freq/路径）
#     if not agent_data:
#         print("⚠️ 未加载到任何RL数据。可能原因：1) 训练步数未达到 eval_freq，未生成 *_data.pkl；"
#               " 2) 数据仍在 runs/<run_id>/<Algo>/artifacts，但未被检索到；3) 权限/路径错误。")
#
#     save_dir = "./results_outputs"
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 2. 绘制第一张图：总成本 vs. Baseline成本 (纯运营成本对比)
#     fig1, ax1 = plt.subplots(figsize=(14, 9))
#     if baseline_cost is not None:
#         ax1.axhline(y=baseline_cost, color='r', linestyle='--', label=f"Baseline 纯运营成本 ({baseline_cost:.2f})")
#
#     all_costs = [baseline_cost] if baseline_cost is not None else []
#     for name, data in agent_data.items():
#         ax1.plot(data['steps'], data['costs'], marker='o', linestyle='-', label=f"{name} 总成本")
#         all_costs.extend(data['costs'])
#
#     ax1.set_title("RL Agent 训练过程【纯运营成本】收敛对比图", fontsize=18)
#     ax1.set_xlabel("训练步数", fontsize=14)
#     ax1.set_ylabel("评估回合总成本 (元)", fontsize=14)
#     if agent_data: ax1.legend(fontsize=12)
#     ax1.grid(True, linestyle=':')
#     if all_costs:
#         min_val, max_val = np.nanmin(all_costs), np.nanmax(all_costs)
#         padding = (max_val - min_val) * 0.1 if max_val > min_val else 10.0
#         ax1.set_ylim(min_val - padding, max_val + padding)
#
#     cost_plot_path = os.path.join(save_dir, "final_training_cost_comparison.png")
#     fig1.savefig(cost_plot_path)
#     print(f"\n✅ 纯成本对比图已保存至: {os.path.abspath(cost_plot_path)}")
#     plt.close(fig1)
#
#     # 3. 绘制第二张图：-总奖励 vs. Baseline总目标值 (综合目标对比)
#     fig2, ax2 = plt.subplots(figsize=(14, 9))
#     if baseline_objective is not None:
#         ax2.axhline(
#             y=baseline_objective,
#             color='r',
#             linestyle='--',
#             label=f"Baseline 总目标值 ({baseline_objective:.2f})"
#         )
#
#     all_obj_vals = [baseline_objective] if baseline_objective is not None else []
#     for name, data in agent_data.items():
#         # 用未缩放奖励来对齐目标函数
#         negative_rewards_unscaled = -np.array(data['rewards_unscaled'])
#         ax2.plot(
#             data['steps'],
#             negative_rewards_unscaled,
#             marker='x',
#             linestyle='-',
#             label=f"{name} (-总奖励_未缩放)"
#         )
#         all_obj_vals.extend(negative_rewards_unscaled)
#
#     ax2.set_title("RL Agent (-总奖励) 与 Baseline (总目标值) 对比图", fontsize=18)
#     ax2.set_xlabel("训练步数", fontsize=14)
#     ax2.set_ylabel("目标函数值 (越低越好)", fontsize=14)
#     if agent_data: ax2.legend(fontsize=12)
#     ax2.grid(True, linestyle=':')
#     if all_obj_vals:
#         min_val, max_val = np.nanmin(all_obj_vals), np.nanmax(all_obj_vals)
#         padding = (max_val - min_val) * 0.1 if max_val > min_val else 10.0
#         ax2.set_ylim(min_val - padding, max_val + padding)
#
#     reward_plot_path = os.path.join(save_dir, "final_training_objective_comparison.png")
#     fig2.savefig(reward_plot_path)
#     print(f"✅ 综合目标对比图已保存至: {os.path.abspath(reward_plot_path)}")
#     plt.close(fig2)
def plot_curves():
    """
    (绘图模式)
    加载所有保存的成本/奖励数据，并绘制：
    1）每个算法单条的成本/目标函数对比图；
    2）多随机种子下的均值 ± 标准差阴影图（如果检测到 seed_* 结构）。
    """
    print("\n" + "=" * 50 + "\n--- 启动最终训练曲线绘图程序（含多随机种子统计） ---")

    import glob
    import os
    from pathlib import Path
    from config import PATHS

    # ========== 统一的“文件查找器” ==========
    # 在 PATHS["models_dir"] 与 项目根目录/runs 下递归查找文件
    def _find_all(filename: str):
        bases = []
        # 1) models 目录（原有逻辑）
        if "models_dir" in PATHS and os.path.isdir(PATHS["models_dir"]):
            bases.append(PATHS["models_dir"])
        # 2) runs 目录（过夜脚本产物归档处）
        runs_dir = Path(__file__).resolve().parent / "runs"
        if runs_dir.is_dir():
            bases.append(str(runs_dir))

        cands = []
        for base in bases:
            cands.extend(glob.glob(os.path.join(base, "**", filename), recursive=True))
        return cands

    def _find_latest(filename: str) -> str:
        cands = _find_all(filename)
        if not cands:
            return ""
        return max(cands, key=os.path.getmtime)

    # ========== 按智能体名找单个 / 多个 data.pkl ==========
    def _find_agent_pkl(agent_name: str) -> str:
        """
        优先在 runs 目录里找该 agent 的最新 data.pkl（用于多随机种子）；
        若 runs 里没有，则退回到全局最新。
        """
        filename = f"{agent_name}_data.pkl"
        all_cands = _find_all(filename)
        if not all_cands:
            return ""
        runs_cands = [p for p in all_cands if os.path.sep + "runs" + os.path.sep in p]
        if runs_cands:
            return max(runs_cands, key=os.path.getmtime)
        return max(all_cands, key=os.path.getmtime)

    def _find_all_agent_pkls(agent_name: str):
        """
        返回当前最新一批 runs/<run_id>/... 下该智能体的所有 <agent_name>_data.pkl，
        用于多随机种子聚合。
        若找不到 seed_* 结构，则退化为只返回最新的那个文件。
        """
        first = _find_agent_pkl(agent_name)
        if not first:
            return []

        p = Path(first).resolve()

        # 尝试向上找到 seed_* 目录（结构：runs/<run_id>/seed_x/<Algo>/artifacts/...）
        run_dir = None
        for parent in p.parents:
            if parent.name.startswith("seed_"):
                run_dir = parent.parent  # seed_x 的上一级就是 run_id 目录
                break

        if run_dir is None or not run_dir.exists():
            # 没有 seed_* 结构，默认只用单文件
            return [first]

        pattern = str(run_dir / "**" / f"{agent_name}_data.pkl")
        files = glob.glob(pattern, recursive=True)
        files = sorted(set(files))
        return files or [first]

    # 要绘制的智能体（与你训练时 CostCurveCallback 的 agent_name 一致）
    agent_names = ["SAC_Two_Stage", "DDPG_Two_Stage", "TD3_Two_Stage", "PPO_Two_Stage"]
    agents_to_plot = {name: _find_agent_pkl(name) for name in agent_names}

    # baseline 文件（谁先更新都行，取最新时间）
    baseline_cost_file = _find_latest("baseline_cost.pkl")
    baseline_objective_file = _find_latest("baseline_objective.pkl")

    # 1. 加载Baseline数据
    try:
        with open(baseline_cost_file, 'rb') as f:
            baseline_cost = pickle.load(f)
        print(f"成功加载Baseline纯运营成本: {baseline_cost:.2f}  @ {baseline_cost_file}")
    except Exception:
        print(f"警告: 未找到Baseline成本文件: {baseline_cost_file}")
        baseline_cost = None

    try:
        with open(baseline_objective_file, 'rb') as f:
            baseline_objective = pickle.load(f)
        print(f"成功加载Baseline总目标值: {baseline_objective:.2f}  @ {baseline_objective_file}")
    except Exception:
        print(f"警告: 未找到Baseline目标值文件: {baseline_objective_file}")
        baseline_objective = None

    # 2. 加载每个算法的“单条曲线”数据（最新的一份）
    agent_data = {}
    for agent_name, data_file in agents_to_plot.items():
        if not data_file:
            print(f"警告: 未找到 {agent_name} 的数据文件，将跳过该模型。")
            continue
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            # 兼容旧版本：没有 unscaled 时，就用 scaled 顶上
            if 'rewards_unscaled' not in data and 'rewards' in data:
                data['rewards_unscaled'] = data['rewards']
            agent_data[agent_name] = data
            print(f"成功加载 {agent_name} 的数据: {data_file}")
        except Exception as e:
            print(f"警告: 打开 {agent_name} 的数据文件失败({data_file})：{e}")

    if not agent_data:
        print("⚠️ 未加载到任何RL数据。可能原因：1) 训练步数未达到 eval_freq，未生成 *_data.pkl；"
              " 2) 数据仍在 runs/<run_id>/<Algo>/artifacts，但未被检索到；3) 权限/路径错误。")

    # 3. 尝试做“多随机种子聚合”
    # multi_seed_data 结构：{agent_name: {steps, cost_mean, cost_std, obj_mean, obj_std, num_runs}}
    multi_seed_data = {}

    for agent_name in agent_names:
        all_pkls = _find_all_agent_pkls(agent_name)
        # 如果只有 1 个文件，就没必要画阴影；跳过
        unique_pkls = sorted(set(all_pkls))
        if len(unique_pkls) <= 1:
            continue

        runs = []
        for pkl_path in unique_pkls:
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                if 'rewards_unscaled' not in data and 'rewards' in data:
                    data['rewards_unscaled'] = data['rewards']
                runs.append(data)
                print(f"[多种子] {agent_name}: 已加载 {pkl_path}")
            except Exception as e:
                print(f"[多种子] {agent_name}: 加载 {pkl_path} 失败: {e}")

        if len(runs) < 2:
            # 有的加载失败，最后只有 1 条，就不做统计了
            continue

        # 以最短长度为准，假设 eval_freq 相同
        lengths = [len(r['steps']) for r in runs if len(r.get('steps', [])) > 0]
        if not lengths:
            continue
        min_len = min(lengths)

        steps = np.array(runs[0]['steps'][:min_len], dtype=float)
        cost_stack = np.stack([np.array(r['costs'][:min_len], dtype=float) for r in runs], axis=0)
        neg_reward_stack = np.stack(
            [
                -np.array(r.get('rewards_unscaled', r['rewards'])[:min_len], dtype=float)
                for r in runs
            ],
            axis=0
        )

        multi_seed_data[agent_name] = {
            'steps': steps,
            'cost_mean': cost_stack.mean(axis=0),
            'cost_std': cost_stack.std(axis=0),
            'obj_mean': neg_reward_stack.mean(axis=0),
            'obj_std': neg_reward_stack.std(axis=0),
            'num_runs': len(runs),
        }

    save_dir = "./results_outputs"
    os.makedirs(save_dir, exist_ok=True)

    # ========== 图 1：单条曲线的【纯运营成本】对比 ==========
    fig1, ax1 = plt.subplots(figsize=(14, 9))
    if baseline_cost is not None:
        ax1.axhline(
            y=baseline_cost,
            color='r',
            linestyle='--',
            label=f"Baseline 纯运营成本 ({baseline_cost:.2f})"
        )

    all_costs = [baseline_cost] if baseline_cost is not None else []
    for name, data in agent_data.items():
        ax1.plot(
            data['steps'],
            data['costs'],
            marker='o',
            linestyle='-',
            label=f"{name} 总成本（单条）"
        )
        all_costs.extend(data['costs'])

    ax1.set_title("RL Agent 训练过程【纯运营成本】收敛对比图（单条曲线）", fontsize=18)
    ax1.set_xlabel("训练步数", fontsize=14)
    ax1.set_ylabel("评估回合总成本 (元)", fontsize=14)
    if agent_data:
        ax1.legend(fontsize=12)
    ax1.grid(True, linestyle=':')
    if all_costs:
        min_val, max_val = np.nanmin(all_costs), np.nanmax(all_costs)
        padding = (max_val - min_val) * 0.1 if max_val > min_val else 10.0
        ax1.set_ylim(min_val - padding, max_val + padding)

    cost_plot_path = os.path.join(save_dir, "final_training_cost_comparison_single.png")
    fig1.savefig(cost_plot_path)
    print(f"✅ 纯成本对比图（单条）已保存至: {os.path.abspath(cost_plot_path)}")
    plt.close(fig1)

    # ========== 图 2：单条曲线的【-总奖励】 vs Baseline 总目标值 ==========
    fig2, ax2 = plt.subplots(figsize=(14, 9))
    if baseline_objective is not None:
        ax2.axhline(
            y=baseline_objective,
            color='r',
            linestyle='--',
            label=f"Baseline 总目标值 ({baseline_objective:.2f})"
        )

    all_obj_vals = [baseline_objective] if baseline_objective is not None else []
    for name, data in agent_data.items():
        negative_rewards_unscaled = -np.array(data['rewards_unscaled'])
        ax2.plot(
            data['steps'],
            negative_rewards_unscaled,
            marker='x',
            linestyle='-',
            label=f"{name} (-总奖励_未缩放，单条)"
        )
        all_obj_vals.extend(negative_rewards_unscaled)

    ax2.set_title("RL Agent (-总奖励) 与 Baseline (总目标值) 对比图（单条曲线）", fontsize=18)
    ax2.set_xlabel("训练步数", fontsize=14)
    ax2.set_ylabel("目标函数值 (越低越好)", fontsize=14)
    if agent_data:
        ax2.legend(fontsize=12)
    ax2.grid(True, linestyle=':')
    if all_obj_vals:
        min_val, max_val = np.nanmin(all_obj_vals), np.nanmax(all_obj_vals)
        padding = (max_val - min_val) * 0.1 if max_val > min_val else 10.0
        ax2.set_ylim(min_val - padding, max_val + padding)

    reward_plot_path = os.path.join(save_dir, "final_training_objective_comparison_single.png")
    fig2.savefig(reward_plot_path)
    print(f"✅ 综合目标对比图（单条）已保存至: {os.path.abspath(reward_plot_path)}")
    plt.close(fig2)

    # ========== 图 3：多随机种子【纯运营成本】均值 ± 标准差 ==========
    if multi_seed_data:
        fig3, ax3 = plt.subplots(figsize=(14, 9))
        if baseline_cost is not None:
            ax3.axhline(
                y=baseline_cost,
                color='r',
                linestyle='--',
                label=f"Baseline 纯运营成本 ({baseline_cost:.2f})"
            )

        all_costs_ms = [baseline_cost] if baseline_cost is not None else []
        for name, agg in multi_seed_data.items():
            steps = agg['steps']
            mean = agg['cost_mean']
            std = agg['cost_std']
            num_runs = agg['num_runs']

            (line,) = ax3.plot(
                steps,
                mean,
                linestyle='-',
                label=f"{name} 总成本均值 (N={num_runs})"
            )
            ax3.fill_between(
                steps,
                mean - std,
                mean + std,
                alpha=0.2,
                color=line.get_color()
            )
            all_costs_ms.extend(mean - std)
            all_costs_ms.extend(mean + std)

        ax3.set_title("RL Agent【纯运营成本】多随机种子均值 ± 标准差", fontsize=18)
        ax3.set_xlabel("训练步数", fontsize=14)
        ax3.set_ylabel("评估回合总成本 (元)", fontsize=14)
        ax3.legend(fontsize=12)
        ax3.grid(True, linestyle=':')
        if all_costs_ms:
            min_val, max_val = np.nanmin(all_costs_ms), np.nanmax(all_costs_ms)
            padding = (max_val - min_val) * 0.1 if max_val > min_val else 10.0
            ax3.set_ylim(min_val - padding, max_val + padding)

        cost_ms_path = os.path.join(save_dir, "final_training_cost_comparison_multiseed.png")
        fig3.savefig(cost_ms_path)
        print(f"✅ 纯成本对比图（多随机种子）已保存至: {os.path.abspath(cost_ms_path)}")
        plt.close(fig3)

        # ========== 图 4：多随机种子【-总奖励】均值 ± 标准差 ==========
        fig4, ax4 = plt.subplots(figsize=(14, 9))
        if baseline_objective is not None:
            ax4.axhline(
                y=baseline_objective,
                color='r',
                linestyle='--',
                label=f"Baseline 总目标值 ({baseline_objective:.2f})"
            )

        all_obj_ms = [baseline_objective] if baseline_objective is not None else []
        for name, agg in multi_seed_data.items():
            steps = agg['steps']
            mean = agg['obj_mean']
            std = agg['obj_std']
            num_runs = agg['num_runs']

            (line,) = ax4.plot(
                steps,
                mean,
                linestyle='-',
                label=f"{name} (-总奖励) 均值 (N={num_runs})"
            )
            ax4.fill_between(
                steps,
                mean - std,
                mean + std,
                alpha=0.2,
                color=line.get_color()
            )
            all_obj_ms.extend(mean - std)
            all_obj_ms.extend(mean + std)

        ax4.set_title("RL Agent (-总奖励) 多随机种子均值 ± 标准差 vs Baseline", fontsize=18)
        ax4.set_xlabel("训练步数", fontsize=14)
        ax4.set_ylabel("目标函数值 (越低越好)", fontsize=14)
        ax4.legend(fontsize=12)
        ax4.grid(True, linestyle=':')
        if all_obj_ms:
            min_val, max_val = np.nanmin(all_obj_ms), np.nanmax(all_obj_ms)
            padding = (max_val - min_val) * 0.1 if max_val > min_val else 10.0
            ax4.set_ylim(min_val - padding, max_val + padding)

        obj_ms_path = os.path.join(save_dir, "final_training_objective_comparison_multiseed.png")
        fig4.savefig(obj_ms_path)
        print(f"✅ 综合目标对比图（多随机种子）已保存至: {os.path.abspath(obj_ms_path)}")
        plt.close(fig4)
    else:
        print("ℹ️ 未检测到多于 1 个的随机种子数据文件，多随机种子阴影图将不会生成。")



if __name__ == '__main__':
    plot_curves()