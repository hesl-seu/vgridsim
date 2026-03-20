# file: simulation_runner.py
import sys
import time
import io
import traceback
import os
import json
import glob
import importlib.util
from PySide6.QtCore import QObject, Signal, Slot
import pandas as pd

# 导入stable_baselines3和环境
from stable_baselines3 import PPO, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from power_grid_env import PowerGridEnv

# 导入项目中的核心函数
from config import PATHS, CORE_PARAMS, TRAINING_CONFIG, RL_ENV_CONFIG
from evaluate_agents import evaluate_baseline, evaluate_rl_agent, plot_and_save_results, \
    plot_accumulated_costs, plot_voltage_snapshots, plot_line_flow_snapshots_comparison, \
    plot_aggregated_ev_power, plot_sop_flows, plot_nop_status
from copy import deepcopy
from config import EVALUATION_CONFIG

def discover_rl_algorithms_util():
    """独立的、安全的工具函数，用于发现所有可用的RL算法类。"""
    model_class_registry = {}
    model_class_registry.update({"PPO": PPO, "DDPG": DDPG, "SAC": SAC, "TD3": TD3})
    plugin_dir = "custom_algorithms"
    if os.path.isdir(plugin_dir):
        plugin_files = glob.glob(os.path.join(plugin_dir, "*.py"))
        for plugin_file in plugin_files:
            module_name = os.path.basename(plugin_file)[:-3]
            try:
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'register_algorithm'):
                    info = module.register_algorithm()
                    model_class_registry[info['name']] = info['class']
            except Exception as e:
                print(f"后台警告: 加载自定义算法插件 {plugin_file} 失败: {e}")
    return model_class_registry


class GuiCallback(BaseCallback):
    def __init__(self, worker, total_timesteps, verbose=0):
        super(GuiCallback, self).__init__(verbose)
        self.worker = worker
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if self.worker.is_stopped:
            return False
        if self.n_calls % 100 == 0:
            self.worker.progress_update.emit(self.num_timesteps, self.total_timesteps)
        return True


class Stream(QObject):
    new_text = Signal(str)

    def write(self, text):
        self.new_text.emit(str(text))

    def flush(self):
        pass


class SimulationWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    error = Signal(str)
    progress_update = Signal(int, int)

    def __init__(self, task_type, task_params):
        super().__init__()
        self.task_type = task_type
        self.task_params = task_params
        self.is_stopped = False

    @Slot()
    def request_stop(self):
        self.progress.emit("...接收到终止信号，将在当前步骤完成后停止...")
        self.is_stopped = True

    @Slot()
    def run(self):
        """
        这个函数是后台线程的入口点。所有耗时操作都必须在这里面执行。
        """
        start_time = time.time()
        # 重定向输出流必须在线程内部完成
        sys.stdout = Stream(new_text=self.progress.emit)
        sys.stderr = Stream(new_text=self.error.emit)

        try:
            # 所有耗时操作都在run()函数内部，确保它们在后台线程执行
            self.progress.emit("后台任务启动，正在准备环境...")

            with open(PATHS["gui_settings"], 'r') as f:
                self.gui_params = json.load(f)
            self.update_core_configs()

            if self.task_type == "run_baseline":
                self.run_baseline_task()
            elif self.task_type == "train_rl":
                self.run_training_task()
            elif self.task_type == "evaluate":
                self.run_evaluation_task()
            else:
                raise ValueError(f"未知的任务类型: {self.task_type}")

        except Exception as e:
            self.error.emit(f"任务执行出错: {e}\n{traceback.format_exc()}")
        finally:
            # 恢复标准输出
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self.progress.emit(f"\n任务结束，总耗时: {time.time() - start_time:.2f} 秒。")
            self.finished.emit()

    def update_core_configs(self):
        CORE_PARAMS['grid_model'] = self.gui_params['grid_model']
        CORE_PARAMS['solver'] = self.gui_params['solver']
        CORE_PARAMS['start_hour'] = self.gui_params['start_hour']
        CORE_PARAMS['end_hour'] = self.gui_params['end_hour']
        CORE_PARAMS['step_minutes'] = self.gui_params['step_minutes']
        CORE_PARAMS['distributed_energy']['pv'] = self.gui_params['use_pv']
        CORE_PARAMS['distributed_energy']['wind'] = self.gui_params['use_wind']
        CORE_PARAMS['distributed_energy']['ess'] = self.gui_params['use_ess']
        CORE_PARAMS['sop_nodes_active'] = self.gui_params['use_sop']
        CORE_PARAMS['nop_nodes_active'] = self.gui_params['use_nop']

        # 将EV数据源设置传递给核心参数
        # 我们从 self.gui_params 中获取这个新设置
        # 如果设置不存在 (例如使用旧的 gui_settings.json)，则默认为 'random'
        CORE_PARAMS['ev_data_source'] = self.gui_params.get('ev_data_source', 'random')

        # EV 物理参数
        CORE_PARAMS['ev_params'] = self.gui_params.get(
            'ev_params',
            CORE_PARAMS.get('ev_params', {})
        )

        # 奖励模式
        CORE_PARAMS['reward_mode'] = self.gui_params.get('reward_mode', 'grid_operator')

        # 运营商参数
        CORE_PARAMS['station_operator'] = self.gui_params.get(
            'station_operator',
            CORE_PARAMS.get('station_operator', {})
        )

        # 奖励权重
        gui_rw = self.gui_params.get('reward_weights', {})
        CORE_PARAMS['reward_weights'] = gui_rw

        RL_ENV_CONFIG['reward_weights']['ev_kwh_shortage_penalty'] = gui_rw.get(
            'ev_kwh_shortage_penalty',
            RL_ENV_CONFIG['reward_weights']['ev_kwh_shortage_penalty']
        )
        RL_ENV_CONFIG['reward_weights']['voltage_violation_penalty'] = gui_rw.get(
            'voltage_violation_penalty',
            RL_ENV_CONFIG['reward_weights']['voltage_violation_penalty']
        )
        RL_ENV_CONFIG['reward_weights']['cost_penalty_factor'] = gui_rw.get(
            'cost_penalty_factor',
            RL_ENV_CONFIG['reward_weights']['cost_penalty_factor']
        )
        RL_ENV_CONFIG['penalties']['opendss_failure_penalty'] = gui_rw.get(
            'opendss_failure_penalty',
            RL_ENV_CONFIG['penalties']['opendss_failure_penalty']
        )

        self.progress.emit("核心配置已根据GUI设置更新。")

    def run_training_task(self):
        self.progress.emit("\n--- 开始训练强化学习模型 ---")

        rl_algo_name = self.task_params['rl_algo_name']
        use_two_stage = self.task_params['mode'] == 'two_stage'

        available_algos = discover_rl_algorithms_util()
        model_class = available_algos.get(rl_algo_name)
        if not model_class:
            raise ValueError(f"找不到名为 {rl_algo_name} 的算法类。")

        self.progress.emit(f"选择算法: {rl_algo_name}")
        self.progress.emit(f"运行模式: {'两阶段' if use_two_stage else '单阶段'}")

        # 环境初始化是一个耗时操作，必须在后台线程中执行
        self.progress.emit("正在初始化仿真环境 (这可能需要一些时间)...")
        env = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=use_two_stage)
        self.progress.emit("环境初始化完成。")

        from training_visualizer import CostCurveCallback

        model = model_class('MlpPolicy', env, verbose=1, tensorboard_log=PATHS["tensorboard_logs"])

        total_timesteps = TRAINING_CONFIG["total_timesteps"]
        gui_callback = GuiCallback(self, total_timesteps)

        # 评估环境要与训练配置一致，单独创建一个
        eval_env = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=use_two_stage)

        # 统一命名，保证 training_visualizer 能找到对应目录和文件
        mode_suffix = "two_stage" if use_two_stage else "single_stage"
        model_dir_name = f"best_{rl_algo_name.lower()}_{mode_suffix}"
        save_path = os.path.join(PATHS["models_dir"], model_dir_name)
        os.makedirs(save_path, exist_ok=True)

        agent_name = f"{rl_algo_name.upper()}_{'Two_Stage' if use_two_stage else 'Single_Stage'}"
        cost_callback = CostCurveCallback(
            eval_env=eval_env,
            agent_name=agent_name,
            save_path=save_path,
            eval_freq=int(TRAINING_CONFIG.get("cost_curve_eval_freq", 500))
        )

        # 同时挂进度回调和采集回调
        callbacks = [gui_callback, cost_callback]

        self.progress.emit(f"总训练步数: {total_timesteps}")
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)

        # 训练结束后显式保存一次采集到的数据（防止提前中断无落盘）
        try:
            cost_callback.save_data()
        except Exception as _:
            pass

        # 保存模型
        model.save(os.path.join(save_path, "best_model.zip"))
        self.progress.emit(f"模型已保存至: {save_path}")

        if self.is_stopped:
            self.progress.emit("\n训练被用户手动终止。")
        else:
            self.progress.emit("\n训练完成！")
            mode_suffix = "two_stage" if use_two_stage else "single_stage"
            model_save_name = f"best_{rl_algo_name.replace('_', ' ').title().replace(' ', '_')}_{mode_suffix}"
            save_path = os.path.join(PATHS["models_dir"], model_save_name)
            os.makedirs(save_path, exist_ok=True)
            model.save(os.path.join(save_path, "best_model.zip"))
            self.progress.emit(f"模型已保存至: {save_path}")

    def run_baseline_task(self):
        self.progress.emit("\n--- 开始运行 Baseline (基于求解器) ---")
        use_two_stage = self.task_params['mode'] == 'two_stage'
        self.progress.emit("正在初始化仿真环境 (这可能需要一些时间)...")
        env_for_scene = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=use_two_stage)
        self.progress.emit("环境初始化完成。")
        env_for_scene.reset(seed=0)
        grid_instance = deepcopy(env_for_scene.grid)
        stations_list = env_for_scene.stations_list
        metrics, time_series_data = evaluate_baseline(
            CORE_PARAMS, 0, stations_list, grid_instance, use_two_stage
        )
        if not metrics:
            raise Exception("Baseline 求解失败，请检查日志输出。")
        self.progress.emit("\n--- Baseline 运行结果 ---")
        for key, val in metrics.items():
            if isinstance(val, float):
                self.progress.emit(f"  - {key}: {val:.4f}")
            else:
                self.progress.emit(f"  - {key}: {val}")

    def run_evaluation_task(self):
        self.progress.emit("\n--- 开始评估对比任务 ---")
        use_two_stage = self.task_params['mode'] == 'two_stage'
        selected_algos = self.task_params['selected_algos']
        num_episodes = EVALUATION_CONFIG.get("num_test_episodes", 10)
        self.progress.emit(f"评估模式: {'两阶段' if use_two_stage else '单阶段'}")
        self.progress.emit(f"对比算法: {', '.join(selected_algos)}")
        model_name_to_class = discover_rl_algorithms_util()
        all_results_metrics = []
        self.progress.emit("正在初始化仿真环境 (这可能需要一些时间)...")
        env = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=use_two_stage)
        self.progress.emit("环境初始化完成。")
        for i in range(num_episodes):
            seed = i
            self.progress.emit(f"\n--- 开始评估场景 {i + 1}/{num_episodes} (seed={seed}) ---")
            ts_log_this_seed = {}
            metrics_log_this_seed = {}
            env.reset(seed=seed)
            grid_instance_for_this_seed = env.grid
            stations_list_for_this_seed = env.stations_list
            for algo_name in selected_algos:
                if self.is_stopped:
                    self.progress.emit("评估任务被用户终止。")
                    return
                if algo_name == "Baseline":
                    grid_for_baseline = deepcopy(grid_instance_for_this_seed)
                    metrics, ts_data = evaluate_baseline(CORE_PARAMS, seed, stations_list_for_this_seed,
                                                         grid_for_baseline, use_two_stage)
                else:
                    folder_name = "best_" + algo_name.lower()
                    model_type_key = next((key for key in model_name_to_class if key.lower() in algo_name.lower()),
                                          None)
                    if not model_type_key:
                        self.progress.emit(f"警告: 找不到与 {algo_name} 匹配的算法类，跳过。")
                        continue
                    model_path = os.path.join(PATHS["models_dir"], folder_name, "best_model.zip")
                    if not os.path.exists(model_path):
                        self.progress.emit(f"警告: 找不到模型文件 {model_path}，跳过 {algo_name}。")
                        continue
                    model_class = model_name_to_class[model_type_key]
                    model = model_class.load(model_path, env=env)
                    metrics, ts_data = evaluate_rl_agent(model, env, seed)
                if metrics:
                    metrics['算法'] = algo_name
                    all_results_metrics.append(metrics)
                    ts_log_this_seed[algo_name] = ts_data
                    metrics_log_this_seed[algo_name] = metrics
                    self.progress.emit(f"  - 算法 '{algo_name}' 评估完成。")
            if ts_log_this_seed:
                self.progress.emit("\n正在为当前场景生成可视化报告...")
                plot_and_save_results(ts_log_this_seed, seed, CORE_PARAMS)
                # plot_accumulated_costs(metrics_log_this_seed, ts_log_this_seed, seed, CORE_PARAMS,
                #                        grid_instance_for_this_seed)
                plot_voltage_snapshots(ts_log_this_seed, seed, CORE_PARAMS)
                plot_line_flow_snapshots_comparison(ts_log_this_seed, seed, CORE_PARAMS)
                plot_aggregated_ev_power(ts_log_this_seed, seed, CORE_PARAMS)
                plot_sop_flows(ts_log_this_seed, seed, CORE_PARAMS)
                plot_nop_status(ts_log_this_seed, seed, CORE_PARAMS)
                self.progress.emit("报告生成完毕！")
        if all_results_metrics:
            results_df = pd.DataFrame(all_results_metrics)
            summary = results_df.groupby('算法').mean()
            summary_std = results_df.groupby('算法').std()
            self.progress.emit("\n\n" + "=" * 80)
            self.progress.emit("评估结果汇总 (Mean - 平均值)")
            self.progress.emit("=" * 80)
            self.progress.emit(summary.to_string(float_format="%.4f"))

            self.progress.emit("\n\n" + "=" * 80)
            self.progress.emit("评估结果汇总 (Std Dev - 标准差)")
            self.progress.emit("=" * 80)
            self.progress.emit(summary_std.to_string(float_format="%.4f"))
            self.progress.emit("=" * 80)