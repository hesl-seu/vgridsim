"""
文件: tune_ppo.py
说明: 自动化化 PPO 超参数调优脚本。
      遍历几组核心超参数，训练模型并保存，方便后续对比最优结果。
"""

import os
import traceback
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# 导入你现有的环境和配置
from power_grid_env import PowerGridEnv
from config import PATHS, CORE_PARAMS, TRAINING_CONFIG, load_gui_settings


def make_env(use_two_stage: bool):
    """构造环境：复用你原有的环境构建逻辑"""
    gui_params = load_gui_settings()
    params = {
        "grid_model": gui_params.get("grid_model", CORE_PARAMS.get("grid_model", "ieee33")),
        "solver": gui_params.get("solver", CORE_PARAMS.get("solver", "gurobi")),
        "start_hour": gui_params.get("start_hour", CORE_PARAMS.get("start_hour", 1)),
        "end_hour": gui_params.get("end_hour", CORE_PARAMS.get("end_hour", 24)),
        "step_minutes": gui_params.get("step_minutes", CORE_PARAMS.get("step_minutes", 60)),
        "distributed_energy": CORE_PARAMS.get("distributed_energy", {}),
        "sop_nodes_active": CORE_PARAMS.get("sop_nodes_active", True),
        "nop_nodes_active": CORE_PARAMS.get("nop_nodes_active", True),
        "slack_bus": CORE_PARAMS.get("slack_bus", "b1"),
        "base_power": CORE_PARAMS.get("base_power", 1.0),
        "ev_data_source": gui_params.get("ev_data_source", CORE_PARAMS.get("ev_data_source", "random")),
        "ev_params": gui_params.get("ev_params", CORE_PARAMS.get("ev_params", {})),
        "reward_weights": gui_params.get("reward_weights", CORE_PARAMS.get("reward_weights", {})),
        "reward_mode": gui_params.get("reward_mode", CORE_PARAMS.get("reward_mode", "grid_operator")),
        "station_operator": gui_params.get("station_operator", CORE_PARAMS.get("station_operator", {})),
    }
    return PowerGridEnv(gui_params=params, use_two_stage_flow=use_two_stage)


def run_tuning_experiment():
    """执行 PPO 网格调参实验"""
    # 基础配置
    TOTAL_TIMESTEPS = 100000  # 与论文中保持一致 100000 步
    EVAL_FREQ = 5000
    RANDOM_SEED = 0

    # ==========================================================
    # 步骤 1: 定义要测试的超参数网格
    # 默认值参考: learning_rate=3e-3, ent_coef=0.01, batch_size=512
    # ==========================================================
    hyperparams_grid = {
        "PPO_Default": {"learning_rate": 3e-3, "ent_coef": 0.01, "batch_size": 512},
        "PPO_High_Expl": {"learning_rate": 3e-3, "ent_coef": 0.05, "batch_size": 512},  # 提高探索率
        "PPO_Low_LR": {"learning_rate": 3e-4, "ent_coef": 0.02, "batch_size": 1024}  # 降低学习率，增大批次使更新更稳定
    }

    print("=" * 50)
    print("开始 PPO 超参数调优实验...")
    print("=" * 50)

    for config_name, params in hyperparams_grid.items():
        print(f"\n---> 正在训练配置: {config_name}")
        print(f"参数: {params}")

        # 设置专门的保存路径
        log_dir = os.path.join(PATHS["logs_dir"], f"tuning_{config_name}")
        best_model_dir = os.path.join(PATHS["models_dir"], f"tuning_{config_name}")
        tb_log_dir = os.path.join(PATHS["tensorboard_logs"], f"tuning_{config_name}")

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)

        # 步骤 2: 创建独立的环境
        try:
            train_env_raw = make_env(use_two_stage=True)
            train_env = DummyVecEnv([lambda: Monitor(train_env_raw)])

            eval_env_raw = make_env(use_two_stage=True)
            eval_env = Monitor(eval_env_raw)

            np.random.seed(RANDOM_SEED)
        except Exception as e:
            print(f"环境创建失败: {e}")
            continue

        # 步骤 3: 实例化 PPO，注入调优参数
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=params["learning_rate"],
            ent_coef=params["ent_coef"],
            batch_size=params["batch_size"],
            tensorboard_log=tb_log_dir,
            verbose=0,  # 设为0以减少刷屏
            seed=RANDOM_SEED,
        )

        # 设置评估回调
        eval_cb = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=best_model_dir,
            log_path=log_dir,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=3,
            deterministic=True,
            render=False
        )

        # 步骤 4: 开始训练
        try:
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb)
            print(f"配置 {config_name} 训练完成。最优模型已保存至: {best_model_dir}")
        except Exception as e:
            print(f"训练异常: {e}")
            traceback.print_exc()
        finally:
            train_env.close()
            eval_env.close()


if __name__ == "__main__":
    run_tuning_experiment()