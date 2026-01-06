# ==============================================================================
# 编码助手平台 - 自定义算法插件样例
#
# 文件名: my_brilliant_algo.py
# 算法名: BrilliantAlgo (基于 TD3)
# 描述: 这是一个可以直接运行的插件文件范例，用于测试平台的插件式架构。
# ==============================================================================

# 1. 导入您的算法实现和所有相关依赖
#    您可以导入任何库，包括您自己编写的算法类。
import os
import shutil
import numpy as np

# 导入平台的核心组件
from power_grid_env import PowerGridEnv
from config import CORE_PARAMS, TRAINING_CONFIG, PATHS

# 导入您算法所依赖的基础库 (这里我们使用stable-baselines3)
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


# --- 您的算法核心实现 ---
# 在这个样例中，我们通过继承 TD3 来创建一个新的 "BrilliantAlgo" 类。
# 您可以在这里添加自定义逻辑，例如修改初始化参数、重写训练循环等。
# 对于测试目的，即使只是简单地重命名，也能验证插件机制是否生效。
class BrilliantAlgo(TD3):
    """
    一个自定义算法样例，继承自 stable_baselines3 的 TD3。
    """

    def __init__(self, policy, env, **kwargs):
        # 您可以在这里添加自定义的初始化逻辑
        print("=" * 50)
        print("Initializing BrilliantAlgo - A custom TD3 variant!")
        print("=" * 50)
        # 调用父类的构造函数
        super().__init__(policy, env, **kwargs)

    def learn(self, total_timesteps, callback=None, log_interval=4, tb_log_name="BrilliantAlgo",
              reset_num_timesteps=True, progress_bar=False):
        # 您可以重写 learn 方法以实现自定义的训练循环
        print(f"\n--- Starting the learning process using BrilliantAlgo's custom learn method! ---")
        # 在这个样例中，我们只是简单调用父类的 learn 方法
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)


# ==============================================================================
# 2. 【必须实现】算法注册函数
#    这是平台自动发现您算法的关键。函数名必须是 register_algorithm。
#    它返回一个字典，包含算法的简称 'name' (大写) 和模型加载时所需的 'class'。
# ==============================================================================
def register_algorithm():
    """向平台注册本文件中的算法。"""
    return {
        'name': 'BRILLIANTALGO',  # <-- 为您的算法起一个独特的、大写的简称
        'class': BrilliantAlgo  # <-- 指向您算法的Python类
    }


# ==============================================================================
# 3. 【推荐实现】训练入口
#    这部分代码使得您的插件文件可以被直接运行以启动训练。
# ==============================================================================
if __name__ == '__main__':
    # 3.1. 为您的模型命名，这将决定日志和最终模型的保存文件夹名
    #      (我们在这里使用 'BrilliantAlgo' 以便和上面的类名对应)
    model_display_name = "BrilliantAlgo_Two_Stage"

    # 3.2. 创建环境 (与平台其他部分保持一致)
    print(f"创建环境: {model_display_name}")
    use_two_stage = 'Two_Stage' in model_display_name
    env = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=use_two_stage)
    eval_env = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=use_two_stage)

    # 3.3. 定义回调函数，用于在训练期间评估和保存最佳模型
    log_path = os.path.join(PATHS["logs_dir"], model_display_name)
    # 评估回调
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_path,
                                 log_path=log_path, eval_freq=TRAINING_CONFIG["eval_freq"],
                                 deterministic=True, render=False)
    # 检查点回调 (可选，用于定期保存模型以防训练中断)
    checkpoint_callback = CheckpointCallback(save_freq=TRAINING_CONFIG["checkpoint_freq"],
                                             save_path=log_path,
                                             name_prefix="rl_model")

    # 3.4. 实例化您的模型
    #      在这里传入您算法所需的所有超参数
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = BrilliantAlgo(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=PATHS["tensorboard_logs"],
        learning_rate=0.001,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=256,
        train_freq=(1, "episode"),
        gradient_steps=-1
    )

    # 3.5. 启动训练
    print(f"--- 即将开始训练模型: {model_display_name} ---")
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=[eval_callback, checkpoint_callback],  # 可以传入多个回调
        tb_log_name=model_display_name
    )

    # 3.6. 训练结束后，自动将最佳模型复制到主模型文件夹，供评估脚本发现
    best_model_path = os.path.join(log_path, "best_model.zip")
    final_model_dir = os.path.join(PATHS["models_dir"], f"best_{model_display_name.lower()}")
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, "best_model.zip")

    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, final_model_path)
        print(f"\n--- 训练完成！最佳模型已自动保存至: {final_model_path} ---")
    else:
        print(f"\n--- 训练完成，但未找到最佳模型文件。请检查训练日志。 ---")