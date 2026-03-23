import os
import json

# ==============================================================================
# 1. 基础路径配置
# ------------------------------------------------------------------------------
# 描述: 定义所有项目中需要用到的文件路径。所有路径都基于此文件的位置动态生成，
#      使得项目可以被移动到任何地方而不需要修改路径。
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "data": os.path.join(PROJECT_ROOT, "data"),
    "grid_params_excel": os.path.join(PROJECT_ROOT, "data", "grid_parameters.xlsx"),
    "ev_scenarios_csv": os.path.join(PROJECT_ROOT, "data", "my_ev_scenarios.csv"),
    "logs_dir": os.path.join(PROJECT_ROOT, "logs"),
    "models_dir": os.path.join(PROJECT_ROOT, "models"),
    "results_dir": os.path.join(PROJECT_ROOT, "results_outputs"),
    "tensorboard_logs": os.path.join(PROJECT_ROOT, "tensorboard_logs"),
    "gui_settings": os.path.join(PROJECT_ROOT, "gui_settings.json"),
}
# 允许通过环境变量覆盖关键路径
import os as _os
PATHS["models_dir"]        = _os.getenv("MODELS_DIR",        PATHS["models_dir"])
PATHS["grid_params_excel"] = _os.getenv("GRID_PARAMS_XLSX",  PATHS["grid_params_excel"])
PATHS["ev_scenarios_csv"]  = _os.getenv("EV_SCENARIOS_CSV",  PATHS["ev_scenarios_csv"])
PATHS["results_dir"]       = _os.getenv("RESULTS_DIR",       PATHS.get("results_dir", os.path.join(PROJECT_ROOT, "results_outputs")))
PATHS["logs_dir"]          = _os.getenv("LOGS_DIR",          PATHS["logs_dir"])
PATHS["tensorboard_logs"]  = _os.getenv("TB_LOGS_DIR",       PATHS["tensorboard_logs"])

# ==============================================================================
# 2. 仿真与电网核心参数 (作为GUI的默认值)
# ------------------------------------------------------------------------------
# 描述: 这个字典现在作为GUI启动时的默认参数。实际运行时，平台将优先使用
#      GUI界面上保存的参数。
# ==============================================================================
CORE_PARAMS = {
    # --- 时间设置 ---
    "start_hour": 0,
    "end_hour": 24,
    "step_minutes": 60,

    # --- 配电网模型基础设置 ---
    "grid_model": "ieee33",      # 可选: "ieee33", "ieee69", "ieee123"
    "slack_bus": "b1",          # 松弛节点
    "base_power": 1.0,          # 单位 MVA 

    # --- 配电网组件开关 ---
    "distributed_energy": {
        "pv": True,    # 是否包含光伏
        "wind": True,  # 是否包含风电
        "ess": True    # 是否包含储能系统
    },
    "sop_nodes_active": True,  # 是否包含SOP（软开点）
    "nop_nodes_active": True,  # 是否包含NOP（常开节点）

    # --- 求解器设置  ---
    "solver": "gurobi",         # 可选: "gurobi", "glpk", "cbc", "scip"
    "ev_data_source": "random",
    "reward_mode": "grid_operator",

    "ev_params": {
        "capacity_kwh": 70.0,
        "max_charge_kw": 60.0,
        "max_discharge_kw": 25.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.90
    },

    "reward_weights": {
        "ev_kwh_shortage_penalty": -100.0,
        "voltage_violation_penalty": -100.0,
        "cost_penalty_factor": 1.0,
        "opendss_failure_penalty": -5000.0
    },

    "station_operator": {
        "charge_service_price": 1.20,
        "v2g_subsidy_price": 0.80,
        "include_grid_cost": False,
        "include_generation_cost": True,
        "include_ess_cost": True,
        "include_sop_loss_cost": True,
        "include_penalty_cost": True
    }

}

# ==============================================================================
# 3. Baseline (全局优化) 参数 
# ==============================================================================
BASELINE_PARAMS = {
    "penalty_factors": {
        "ev_not_full_penalty": 100,      # EV未充满惩罚 (元/kWh)
        "slack_power_penalty": 1e7,      # 功率不平衡松弛变量惩罚
        "sop_capacity_penalty": 1e6,     # SOP容量松弛变量惩罚
        "nop_voltage_penalty": 1e6,      # NOP电压松弛变量惩罚
    },
    "ess_degradation_cost": 0.0,         # ESS放电折旧成本 (元/kWh)
}

# ==============================================================================
# 4. 运行模式与评估配置
# ==============================================================================
EVALUATION_CONFIG = {
    "flow_mode": 'two_stage', 
    "num_test_episodes": 1,
    "standards": {
        "voltage_min_pu": 0.90,
        "voltage_max_pu": 1.10,
        "ev_charged_soc_threshold": 0.95,
    }
}

# ==============================================================================
# 5. 强化学习环境与奖励函数配置 
# ==============================================================================
RL_ENV_CONFIG = {
    "reward_weights": {
        "ev_kwh_shortage_penalty": -100.0,
        "voltage_violation_penalty": -100.0,
        "cost_penalty_factor": 1,
    },
    "penalties": {
            "opendss_failure_penalty": -5000.0
    },
    # 缩放因子
    "reward_scale": 0.01,
}

# ==============================================================================
# 6. 训练过程配置 
# ==============================================================================
TRAINING_CONFIG = {
    "total_timesteps": 100000,
    "eval_freq": 240,
    "checkpoint_freq": 10000,
    "cost_curve_eval_freq": 240
}
# ==============================================================================
# RL 算法超参数配置
# ==============================================================================
RL_HYPERPARAMS = {
    "common": {
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "batch_size": 256
    },
    "algo_specific": {
        "PPO": {"clip_range": 0.2, "ent_coef": 0.0},
        "SAC": {"tau": 0.005, "ent_coef": 0.1}, # SAC的ent_coef这里先用浮点数代表初始值
        "DDPG": {"tau": 0.005, "action_noise": 0.1},
        "TD3": {"policy_delay": 2, "target_policy_noise": 0.2}
    }
}
# ==============================================================================
# 7. 自动计算的派生参数
# ==============================================================================
try:
    _total_duration_hours = CORE_PARAMS['end_hour'] - CORE_PARAMS['start_hour']
    if _total_duration_hours <= 0:
        raise ValueError("仿真结束时间必须大于开始时间")

    if 60 % CORE_PARAMS['step_minutes'] != 0:
        raise ValueError("仿真步长必须能被60整除")

    _steps_per_hour = 60 // CORE_PARAMS['step_minutes']
    TIMESTEPS_PER_EPISODE = int(_total_duration_hours * _steps_per_hour)
except (KeyError, ValueError) as e:
    print(f"错误: 自动计算总步数失败 - {e}")
    TIMESTEPS_PER_EPISODE = 24


def load_gui_settings():
    """
    加载GUI设置。如果配置文件存在，则读取；否则，使用config.py中的默认值。
    """
    if os.path.exists(PATHS["gui_settings"]):
        with open(PATHS["gui_settings"], 'r') as f:
            return json.load(f)
    else:
        # 如果文件不存在，则从CORE_PARAMS构建一个GUI配置的子集
        default_settings = {
            "grid_model": CORE_PARAMS["grid_model"],
            "solver": CORE_PARAMS["solver"],
            "start_hour": CORE_PARAMS["start_hour"],
            "end_hour": CORE_PARAMS["end_hour"],
            "step_minutes": CORE_PARAMS["step_minutes"],
            "use_pv": CORE_PARAMS["distributed_energy"]["pv"],
            "use_wind": CORE_PARAMS["distributed_energy"]["wind"],
            "use_ess": CORE_PARAMS["distributed_energy"]["ess"],
            "use_sop": CORE_PARAMS["sop_nodes_active"],
            "use_nop": CORE_PARAMS["nop_nodes_active"],

            "ev_data_source": CORE_PARAMS["ev_data_source"],
            "ev_params": CORE_PARAMS["ev_params"],
            "reward_weights": CORE_PARAMS["reward_weights"],
            "reward_mode": CORE_PARAMS["reward_mode"],
            "station_operator": CORE_PARAMS["station_operator"],


            "rl_common": {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "batch_size": 256
            },
            "rl_specific": {
                "PPO": {"clip_range": 0.2, "ent_coef": 0.0},
                "SAC": {"tau": 0.005, "ent_coef": 0.1},
                "DDPG": {"tau": 0.005, "action_noise": 0.1},
                "TD3": {"policy_delay": 2, "target_policy_noise": 0.2}
            }
        }
        return default_settings

def get_effective_rl_hyperparams(algo_name: str, gui_settings: dict | None = None) -> dict:
    """
    统一计算“最终生效”的 RL 超参数。

    优先级：
    1) GUI 保存的 gui_settings.json
    2) config.py 中的 RL_HYPERPARAMS 默认值

    返回格式：
    {
        "common": {...},
        "specific": {...}
    }
    """
    algo_key = str(algo_name).upper()
    settings = gui_settings if isinstance(gui_settings, dict) else load_gui_settings()

    common_defaults = dict(RL_HYPERPARAMS.get("common", {}))
    specific_defaults = dict(RL_HYPERPARAMS.get("algo_specific", {}).get(algo_key, {}))

    gui_common = dict(settings.get("rl_common", {})) if isinstance(settings, dict) else {}
    gui_specific_all = settings.get("rl_specific", {}) if isinstance(settings, dict) else {}
    gui_specific = dict(gui_specific_all.get(algo_key, {})) if isinstance(gui_specific_all, dict) else {}

    return {
        "common": {**common_defaults, **gui_common},
        "specific": {**specific_defaults, **gui_specific},
    }


