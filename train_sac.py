# import os
# import traceback
# from stable_baselines3 import SAC
# from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv
# from config import CORE_PARAMS, PATHS, TRAINING_CONFIG
# # 导入您的RL环境
# from power_grid_env import PowerGridEnv
# from training_visualizer import CostCurveCallback
#
# def main():
#     """
#     主训练函数
#     """
#     # 1. 直接使用 config 中的参数
#     gui_params = CORE_PARAMS
#     agent_name = "SAC_Two_Stage" # 定义一个智能体名称
#
#     # 2. 基于 config 中的路径构建日志和模型保存路径
#     log_subfolder = agent_name.lower() # e.g., "sac_two_stage"
#     LOG_DIR = os.path.join(PATHS["logs_dir"], log_subfolder)
#     BEST_MODEL_SAVE_PATH = os.path.join(PATHS["models_dir"], f"best_{log_subfolder}")
#     FINAL_MODEL_SAVE_PATH = os.path.join(PATHS["models_dir"], f"{log_subfolder}_final.zip")
#     TENSORBOARD_LOG_DIR = PATHS["tensorboard_logs"]
#
#     # 3. 在任何操作之前，优先创建所有需要的文件夹
#     try:
#         os.makedirs(LOG_DIR, exist_ok=True)
#         os.makedirs(BEST_MODEL_SAVE_PATH, exist_ok=True)
#         os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
#     except OSError as e:
#         print(f"创建文件夹时出错: {e}")
#         return
#
#     # 4. 创建并配置环境
#     try:
#
#
#         print("为训练环境创建实例...")
#         unwrapped_train_env = PowerGridEnv(gui_params=gui_params, use_two_stage_flow=True)
#
#         print("为评估环境创建独立实例...")
#         unwrapped_eval_env = PowerGridEnv(gui_params=gui_params, use_two_stage_flow=True)
#
#
#         print("PowerGridEnv 训练和评估环境创建成功。")
#         print(
#             f"当前潮流模式: {'两阶段 (DistFlow+OpenDSS)' if unwrapped_eval_env.use_two_stage_flow else '单阶段 (仅DistFlow)'}")
#
#         eval_env = Monitor(unwrapped_eval_env)
#         train_env = Monitor(unwrapped_train_env)
#         train_env = DummyVecEnv([lambda: train_env])
#
#     except Exception as e:
#         print(f"创建PowerGridEnv环境时发生错误: {e}")
#         traceback.print_exc()
#         return
#
#     # 配置回调函数 (使用 config 中的值)
#     checkpoint_callback = CheckpointCallback(
#         save_freq=TRAINING_CONFIG["checkpoint_freq"],  # <-- 使用config
#         save_path=LOG_DIR,
#         name_prefix="sac_checkpoint"
#     )
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=BEST_MODEL_SAVE_PATH,
#         log_path=LOG_DIR,
#         eval_freq=TRAINING_CONFIG["eval_freq"],  # <-- 使用config
#         deterministic=True,
#         render=False
#     )
#     cost_callback = CostCurveCallback(
#         eval_env=unwrapped_eval_env,
#         agent_name=agent_name,
#         save_path=BEST_MODEL_SAVE_PATH,
#         eval_freq=TRAINING_CONFIG["cost_curve_eval_freq"]  # <-- 使用config
#     )
#
#     # 创建SAC模型
#     model = SAC(
#         "MlpPolicy",
#         train_env,
#         verbose=1,
#         tensorboard_log=TENSORBOARD_LOG_DIR
#     )
#
#     #开始训练
#     print("\n" + "=" * 40)
#     print("开始SAC模型训练... 您可以随时按 Ctrl+C 来手动中断。")
#     print(f"日志和检查点将保存在: {os.path.abspath(LOG_DIR)}")
#     print(f"最优模型将被保存在: {os.path.abspath(BEST_MODEL_SAVE_PATH)}")
#     print("=" * 40 + "\n")
#
#     try:
#         model.learn(
#             total_timesteps=TRAINING_CONFIG["total_timesteps"],  # <-- 使用config
#             callback=[checkpoint_callback, eval_callback, cost_callback],
#             progress_bar=True,
#             tb_log_name=log_subfolder
#         )
#     except KeyboardInterrupt:
#         print("\n训练被用户手动中断。")
#     except Exception as e:
#         print(f"\n训练过程中发生未知错误: {e}")
#         traceback.print_exc()
#     finally:
#         print("正在执行最终清理和收尾工作...")
#         cost_callback.save_data()
#         print("正在关闭日志记录器以确保数据被保存...")
#         # 这一行是关键，它会强制将所有缓存的日志数据写入硬盘
#         model.logger.close()
#         print(f"正在保存最终模型到: {os.path.abspath(FINAL_MODEL_SAVE_PATH)}")
#         model.save(FINAL_MODEL_SAVE_PATH)
#         print("最终模型已保存。")
#         print(f"训练过程中找到的最优模型已保存在 '{os.path.abspath(BEST_MODEL_SAVE_PATH)}' 文件夹中。")
#         train_env.close()
#         eval_env.close()
#
#
# if __name__ == '__main__':
#     main()
# -*- coding: utf-8 -*-
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
