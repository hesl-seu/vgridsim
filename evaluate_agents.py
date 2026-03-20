import pandas as pd
import matplotlib.pyplot as plt
import os
from power_grid_env import PowerGridEnv
from baseline import solve_baseline
from grid_model import create_grid, load_electricity_price, load_station_info
import traceback
import matplotlib as mpl
from visualization import plot_ev_spot_powers, plot_voltage_snapshots, plot_line_flow_snapshots_comparison
from fpowerkit.soldss import OpenDSSSolver
from fpowerkit.solbase import GridSolveResult
from two_stage_powerflow import fix_bus_voltage_limits, update_grid_from_model
from fpowerkit import Generator
from copy import deepcopy
from config import CORE_PARAMS, EVALUATION_CONFIG, PATHS
import importlib.util
import glob
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import numpy as np
import re

def calc_station_operator_step_metrics(ev_power_kw, price_t, step_minutes, ev_params, station_cfg, info):
    import numpy as np

    ev_power_kw = np.asarray(ev_power_kw, dtype=float)
    charge_kw = np.clip(ev_power_kw, 0.0, None)
    discharge_kw = np.clip(-ev_power_kw, 0.0, None)

    step_h = step_minutes / 60.0
    pi_uc = float(station_cfg.get("charge_service_price", 1.20))
    pi_ud = float(station_cfg.get("v2g_subsidy_price", 0.80))
    eta_c = max(ev_params.charge_efficiency, 1e-6)
    eta_d = max(ev_params.discharge_efficiency, 1e-6)

    charge_profit = float(np.sum((pi_uc - price_t) * charge_kw * eta_c * step_h))
    discharge_profit = float(np.sum((price_t - pi_ud) * discharge_kw / eta_d * step_h))
    gross_profit = charge_profit + discharge_profit

    extra_cost = 0.0
    if station_cfg.get("include_grid_cost", False):
        extra_cost += info.get("grid_purchase_cost", 0.0)
    if station_cfg.get("include_generation_cost", True):
        extra_cost += info.get("generation_cost", 0.0)
    if station_cfg.get("include_ess_cost", True):
        extra_cost += info.get("ess_discharge_cost", 0.0)
    if station_cfg.get("include_sop_loss_cost", True):
        extra_cost += info.get("sop_loss_cost", 0.0)

    penalty_cost = 0.0
    if station_cfg.get("include_penalty_cost", True):
        penalty_cost += -info.get("voltage_penalty_unscaled", 0.0)
        penalty_cost += -info.get("opendss_failure_penalty_unscaled", 0.0)
        penalty_cost += -info.get("ev_shortage_penalty_unscaled", 0.0)

    net_profit = gross_profit - extra_cost - penalty_cost

    return {
        "charge_service_profit": charge_profit,
        "v2g_spread_profit": discharge_profit,
        "station_gross_profit": gross_profit,
        "station_extra_cost": extra_cost,
        "station_penalty_cost": penalty_cost,
        "station_net_profit": net_profit,
    }
# 此函数，用于发现和加载所有算法插件
def discover_and_load_algorithms():
    """
    扫描 custom_algorithms 和 stable_baselines3 目录，动态加载所有模型类。
    """
    model_class_registry = {}

    # 1. 加载 stable-baselines3 的内置算法 (作为基础)
    from stable_baselines3 import PPO, DDPG, SAC, TD3
    model_class_registry.update({"PPO": PPO, "DDPG": DDPG, "SAC": SAC, "TD3": TD3})

    # 2. 扫描 custom_algorithms 目录下的所有 python 文件
    plugin_dir = "custom_algorithms"
    plugin_files = glob.glob(os.path.join(plugin_dir, "*.py"))

    for plugin_file in plugin_files:
        module_name = os.path.basename(plugin_file)[:-3]
        try:
            # 动态导入插件模块
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 检查并调用注册函数
            if hasattr(module, 'register_algorithm'):
                registration_info = module.register_algorithm()
                algo_name = registration_info['name']
                algo_class = registration_info['class']
                model_class_registry[algo_name] = algo_class
                print(f"  [Plugin Loader] 成功加载自定义算法: '{algo_name}' from {plugin_file}")
            else:
                print(f"  [Plugin Loader] 警告: 文件 {plugin_file} 不是一个有效的插件 (缺少 register_algorithm 函数)。")
        except Exception as e:
            print(f"  [Plugin Loader] 错误: 加载插件 {plugin_file} 失败: {e}")

    return model_class_registry

def run_baseline_stage_two(grid: object, baseline_data: object, gui_params: object) -> object:
    """
    - 修正: 在调用print_power_audit之前，先为grid对象加载AUDIT_TIMESTEP的状态。
    """
    AUDIT_TIMESTEP = 5
    print("\n" + "=" * 20 + " Baseline Stage 2: OpenDSS 精确计算开始 " + "=" * 17)
    sb_mva = grid.SB
    step_minutes = gui_params['step_minutes']
    step_duration_h = step_minutes / 60.0
    price = load_electricity_price(gui_params=gui_params)
    time_steps = int((gui_params['end_hour'] - gui_params['start_hour']) * (60 // step_minutes))
    source_bus_id = gui_params.get('slack_bus', 'b1')

    VIRTUAL_GEN_ID = 'gen_for_slack_bus'
    slack_generator = grid.Gen(VIRTUAL_GEN_ID) if VIRTUAL_GEN_ID in grid.GenNames else None
    if not slack_generator:
        slack_generator = Generator(id=VIRTUAL_GEN_ID, busid=source_bus_id, pmax_pu=9999, pmin_pu=-9999, qmax_pu=9999,
                                    qmin_pu=-9999, costA=0, costB=0, costC=0)
        grid.AddGen(slack_generator)

    sop_gen_map = {}
    if hasattr(grid, 'SOPs'):
        for sop_id, sop in grid.SOPs.items():
            gen1_id = f"gen_for_{sop_id}_bus1";
            gen2_id = f"gen_for_{sop_id}_bus2"
            if gen1_id in grid.GenNames: grid.DelGen(gen1_id)
            if gen2_id in grid.GenNames: grid.DelGen(gen2_id)
            gen1 = Generator(id=gen1_id, busid=sop.Bus1, pmax_pu=9999, pmin_pu=-9999, qmax_pu=9999, qmin_pu=-9999,
                             costA=0, costB=0, costC=0)
            gen2 = Generator(id=gen2_id, busid=sop.Bus2, pmax_pu=9999, pmin_pu=-9999, qmax_pu=9999, qmin_pu=-9999,
                             costA=0, costB=0, costC=0)
            grid.AddGen(gen1);
            grid.AddGen(gen2)
            sop_gen_map[sop_id] = (gen1, gen2)

    voltages_stage1_data = baseline_data.get('bus_voltages', {})
    total_generation_cost = baseline_data.get('generation_cost', 0)
    total_ess_discharge_cost = baseline_data.get('ess_discharge_cost', 0)
    fix_bus_voltage_limits(grid)
    opendss_solver = OpenDSSSolver(grid, source_bus=source_bus_id)

    spot_to_bus_map = {}
    spot_counter = 0
    stations_info = load_station_info()
    # 当前电网里实际存在的母线名
    valid_buses = set(getattr(grid, "_bnames", [b.ID for b in grid.Buses]))

    for i, info in enumerate(stations_info):
        bus_id = str(info['Bus_ID'])
        station_id = info.get('Station_ID', f"ST_{i}")

        for j in range(info['Num_Spots']):
            global_spot_idx = spot_counter + j

            # 只有当母线在当前网架中存在时，才建立映射
            if bus_id in valid_buses:
                spot_to_bus_map[global_spot_idx] = bus_id
            else:
                # 这里可以打个提示，方便在控制台看
                print(f"警告：充电站 {station_id} 的母线 {bus_id} 不在当前配电网中，跳过这些车位的映射。")

        # 注意：不管有没有映射，都要累加 spot_counter，以保持索引和 Baseline 一致
        spot_counter += info['Num_Spots']


    # 在调用审计函数之前，专门为 AUDIT_TIMESTEP 准备一次电网状态
    if AUDIT_TIMESTEP is not None and AUDIT_TIMESTEP < time_steps:
        # 1. 更新所有常规设备在 t=AUDIT_TIMESTEP 的状态
        update_grid_from_model(grid, baseline_data, AUDIT_TIMESTEP)

        # 2. 更新SOP虚拟发电机在 t=AUDIT_TIMESTEP 的状态
        sop_flows_data = baseline_data.get('sop_flows', {})
        for sop_id, flow_data in sop_flows_data.items():
            if sop_id in sop_gen_map:
                gen1, gen2 = sop_gen_map[sop_id]
                p_transfer = flow_data['P1'][AUDIT_TIMESTEP]
                q_transfer = flow_data['Q1'][AUDIT_TIMESTEP]

                loss_transfer = flow_data['Loss'][AUDIT_TIMESTEP]
                gen1._p, gen1._q = -p_transfer, -q_transfer
                gen2._p, gen2._q = p_transfer - loss_transfer, q_transfer  # 注入端功率要减去损耗




    # --- 主仿真循环开始 ---
    voltages_stage2_log = []
    line_powers_stage2_log = []
    step_costs_log = []
    total_precise_grid_cost = 0.0
    total_loss_W = 0.0
    stage2_inflow_p_log_pu = []

    for t in range(time_steps):
        # 在循环内部，为当前时间步t更新状态
        update_grid_from_model(grid, baseline_data, t)

        sop_flows_data = baseline_data.get('sop_flows', {})
        for sop_id, flow_data in sop_flows_data.items():
            if sop_id in sop_gen_map:
                gen1, gen2 = sop_gen_map[sop_id]
                p_transfer = flow_data['P1'][t]
                q_transfer = flow_data['Q1'][t]
                loss_transfer = flow_data['Loss'][t]
                gen1._p, gen1._q = -p_transfer, -q_transfer
                gen2._p, gen2._q = p_transfer - loss_transfer, q_transfer  # 注入端功率要减去损耗
         

        spot_powers_data = baseline_data.get('spot_powers', {})
        bus_ev_load_pu = {b.ID: 0.0 for b in grid.Buses}
        for spot_id_num, power_list in spot_powers_data.items():
            # 只要功率不是0 (无论正负)，都进行处理
            if t < len(power_list) and abs(power_list[t]) > 1e-6:
                bus_id = spot_to_bus_map.get(spot_id_num)
                # 既要求有 bus_id，也要求这个 bus_id 真的是当前网架的一个母线
                if bus_id and bus_id in bus_ev_load_pu:
                    # 充电(正)和V2G(负)都会被累加
                    bus_ev_load_pu[bus_id] += power_list[t]

        original_load_funcs = {}
        try:
            for bus_id, ev_load in bus_ev_load_pu.items():
                # 这样负的 ev_load (V2G) 也能被应用
                if abs(ev_load) > 1e-6:
                    bus = grid.Bus(bus_id)
                    original_func = bus.Pd
                    original_load_funcs[bus_id] = original_func
                    # V2G(负值)会减小Pd, 充电(正值)会增加Pd
                    bus.Pd = lambda time, _of=original_func, _el=ev_load: _of(time) + _el
            time_in_seconds = t * step_minutes * 60
            opendss_result, opendss_value = opendss_solver.solve(time_in_seconds)
        finally:
            for bus_id, original_func in original_load_funcs.items():
                grid.Bus(bus_id).Pd = original_func

        step_cost_t = 0
        if opendss_result == GridSolveResult.OK:
            total_loss_W += opendss_value
            voltages_stage2_log.append({bus.ID: bus.V for bus in grid.Buses if bus.V is not None})
            line_powers_stage2_log.append({line.ID: line.P for line in grid.Lines if line.P is not None})
            precise_inflow_pu = slack_generator.P if slack_generator.P is not None else 0
            stage2_inflow_p_log_pu.append(precise_inflow_pu)
            precise_inflow_pu = max(0.0, precise_inflow_pu)  # 禁止负功率计入购电成本
            precise_grid_cost_step = price[t] * precise_inflow_pu * sb_mva * 1000 * step_duration_h
            total_precise_grid_cost += precise_grid_cost_step
            step_cost_t += precise_grid_cost_step
        else:
            print(f"  - 警告: 时间步 {t} OpenDSS 求解失败")
            stage2_inflow_p_log_pu.append(np.nan)
            voltages_stage2_log.append({})
            line_powers_stage2_log.append({})
            step_cost_t = np.nan
        step_costs_log.append(step_cost_t)

    stage1_inflow_p_log_pu = baseline_data.get('grid_inflow_p', [0] * time_steps)


    metrics = {
        "购电成本": total_precise_grid_cost,
        "发电成本": total_generation_cost,
        "SOP损耗成本": baseline_data.get('sop_loss_cost', 0),
        "ESS放电成本": total_ess_discharge_cost,
        "精确总网损(kW)": total_loss_W / 1000.0
    }
    metrics["总成本"] = sum(m for k, m in metrics.items() if '网损' not in k)

    stage1_total_obj = baseline_data.get('objective_value', 0)
    stage1_purchase_cost = baseline_data.get('grid_purchase_cost', 0)
    final_hybrid_obj_value = (stage1_total_obj - stage1_purchase_cost) + total_precise_grid_cost
    metrics["总目标值"] = final_hybrid_obj_value

    standards = EVALUATION_CONFIG["standards"]
    violations = sum(1 for v_dict in voltages_stage2_log for v in v_dict.values()
                     if not (standards["voltage_min_pu"] <= v <= standards["voltage_max_pu"]))
    total_checks = sum(len(v_dict) for v_dict in voltages_stage2_log)
    metrics["电压合格率(%)"] = (1 - violations / total_checks) * 100 if total_checks > 0 else 100.0

    satisfied_count = baseline_data.get('charged_ev_count', 0)
    total_count = baseline_data.get('total_ev_count', 0)
    metrics["EV充电满足率(%)"] = (satisfied_count / total_count) * 100 if total_count > 0 else 100


    def format_log_data(log):
        data_final = {}
        if log:
            all_keys = set().union(*(d.keys() for d in log if d))
            for key in all_keys: data_final[key] = [step.get(key, np.nan) for step in log]
        return data_final

    time_series_data = {
        "voltages_data_stage1": voltages_stage1_data,
        "voltages_data_stage2": format_log_data(voltages_stage2_log),
        "line_powers_data_stage1": baseline_data.get('line_powers', {}),
        "line_powers_data_stage2": format_log_data(line_powers_stage2_log),
        "step_costs": step_costs_log,
        "raw_baseline_data": baseline_data,
    }
    return metrics, time_series_data

def evaluate_rl_agent(model, env, seed):
    """
    评估单个RL智能体，并返回标准化的指标和时序数据。
    - 修正了数据日志记录和返回结构，确保函数总能正确返回。
    """
    obs, info = env.reset(seed=seed)
    metrics = {
        "购电成本": 0.0,
        "发电成本": 0.0,
        "SOP损耗成本": 0.0,
        "ESS放电成本": 0.0,
        "环境累计奖励(缩放后)": 0.0,
        "环境累计奖励(未缩放)": 0.0,
    }

    reward_mode = env.params.get("reward_mode", "grid_operator")
    station_cfg = env.params.get("station_operator", {})

    if reward_mode == "station_operator":
        metrics.update({
            "充电服务价差收益": 0.0,
            "V2G价差收益": 0.0,
            "运营商毛收益": 0.0,
            "运营商附加成本": 0.0,
            "运营商惩罚成本": 0.0,
            "运营商净收益": 0.0,
        })

    # 初始化所有日志记录器
    voltages_stage1_log, voltages_stage2_log = [], []
    line_powers_stage1_log, line_powers_stage2_log = [], []
    total_loss_kW = 0.0
    pvw_pu_log, ess_soc_log, spot_kw_log = [], [], []
    step_costs_log = []
    sop_p_log, sop_q_log, nop_status_log = [], [], []
    total_episode_reward = 0.0

    terminated = False
    while not terminated:
        current_ess_soc = [ess.SOC * 100 for ess in env.ess_list] if env.ess_list else []
        ess_soc_log.append(current_ess_soc)

        action, _ = model.predict(obs, deterministic=True)

        action_idx = env.total_spots + len(env.ess_list) + len(env.pvw_list)
        sop_p_action = action[action_idx: action_idx + len(env.sop_list)]
        action_idx += len(env.sop_list)
        sop_q_action = action[action_idx: action_idx + len(env.sop_list)]
        action_idx += len(env.sop_list)
        nop_action = action[action_idx: action_idx + len(env.nop_list)]

        sop_p_log.append([a * sop.PMax for a, sop in zip(sop_p_action, env.sop_list)])
        sop_q_log.append([a * sop.QMax for a, sop in zip(sop_q_action, env.sop_list)])
        nop_status_log.append(np.round(np.clip(nop_action, 0, 1)).astype(int))

        obs, reward, terminated, truncated, info = env.step(action)

        total_episode_reward += reward
        metrics["环境累计奖励(缩放后)"] += reward
        metrics["环境累计奖励(未缩放)"] += info.get("reward_unscaled", reward)

        grid_cost = info.get('grid_purchase_cost', 0)
        gen_cost = info.get('generation_cost', 0)
        sop_cost = info.get('sop_loss_cost', 0)
        ess_cost = info.get('ess_discharge_cost', 0)

        metrics["购电成本"] += grid_cost
        metrics["发电成本"] += gen_cost
        metrics["SOP损耗成本"] += sop_cost
        metrics["ESS放电成本"] += ess_cost
        physical_step_cost = grid_cost + gen_cost + sop_cost + ess_cost
        step_costs_log.append(physical_step_cost)

        if reward_mode == "station_operator":
            step_station = {}
            if "station_net_profit" in info:
                step_station = {
                    "charge_service_profit": info.get("charge_service_profit", 0.0),
                    "v2g_spread_profit": info.get("v2g_spread_profit", 0.0),
                    "station_gross_profit": info.get("station_gross_profit", 0.0),
                    "station_extra_cost": info.get("station_extra_cost", 0.0),
                    "station_penalty_cost": info.get("station_penalty_cost", 0.0),
                    "station_net_profit": info.get("station_net_profit", 0.0),
                }
            else:
                price_t = env.price[env.current_step - 1]
                step_station = calc_station_operator_step_metrics(
                    ev_power_kw=info.get("ev_power_kw", []),
                    price_t=price_t,
                    step_minutes=env.params["step_minutes"],
                    ev_params=env.ev_params,
                    station_cfg=station_cfg,
                    info=info
                )

            metrics["充电服务价差收益"] += step_station["charge_service_profit"]
            metrics["V2G价差收益"] += step_station["v2g_spread_profit"]
            metrics["运营商毛收益"] += step_station["station_gross_profit"]
            metrics["运营商附加成本"] += step_station["station_extra_cost"]
            metrics["运营商惩罚成本"] += step_station["station_penalty_cost"]
            metrics["运营商净收益"] += step_station["station_net_profit"]

        # 分别记录两阶段的电压和潮流
        voltages_stage1_log.append(info.get('voltages_stage1', {}))
        voltages_stage2_log.append(info.get('voltages_stage2', {}))
        line_powers_stage1_log.append(info.get('line_powers_stage1', {}))
        line_powers_stage2_log.append(info.get('line_powers_stage2', {}))

        # 累加精确网损
        total_loss_kW += info.get('opendss_loss_W', 0.0) / 1000.0

        pvw_pu_log.append(info.get('pvw_power_pu', [0] * len(env.pvw_list)))
        spot_kw_log.append(info.get('ev_power_kw', [0] * env.total_spots))

    final_ess_soc = [ess.SOC * 100 for ess in env.ess_list] if env.ess_list else []
    ess_soc_log.append(final_ess_soc)

    metrics["总成本"] = sum(metrics[k] for k in ["购电成本", "发电成本", "SOP损耗成本", "ESS放电成本"])
    if reward_mode == "station_operator":
        metrics["总目标值"] = metrics["运营商净收益"]
    else:
        metrics["总目标值"] = -metrics["环境累计奖励(未缩放)"]
    metrics["精确总网损(kW)"] = total_loss_kW


    # --- EV充电满足率计算 ---
    all_sessions = []
    for station in env.stations_list:
        all_sessions.extend(station.daily_sessions)
    total_evs_in_scenario = len(all_sessions)
    ev_satisfaction_count = 0
    if total_evs_in_scenario > 0:
        for session in all_sessions:
            departure_step = int(
                round((session.departure_hour - env.params['start_hour']) * (60 // env.params['step_minutes'])))
            departure_step = min(departure_step, env.total_timesteps)
            final_soc = env.ev_boc[session.spot_id, departure_step]
            standards = EVALUATION_CONFIG["standards"]
            if final_soc >= standards["ev_charged_soc_threshold"]:
                ev_satisfaction_count += 1
        metrics["EV充电满足率(%)"] = (ev_satisfaction_count / total_evs_in_scenario) * 100
    else:
        metrics["EV充电满足率(%)"] = 100.0

    # --- 电压合格率计算 ---
    violations = sum(1 for v_dict in voltages_stage2_log for v in v_dict.values()
                     if not (standards["voltage_min_pu"] <= v <= standards["voltage_max_pu"]))  # <-- 修改
    total_checks = sum(len(v_dict) for v_dict in voltages_stage2_log)
    metrics["电压合格率(%)"] = (1 - violations / total_checks) * 100 if total_checks > 0 else 100.0

    # --- 数据格式化 ---
    def format_log_data(log):
        data_final = {}
        if log:
            all_keys = set().union(*(d.keys() for d in log if d))
            for key in all_keys:
                data_final[key] = [step.get(key, np.nan) for step in log]
        return data_final

    voltages_s1_final = format_log_data(voltages_stage1_log)
    voltages_s2_final = format_log_data(voltages_stage2_log)
    line_powers_s1_final = format_log_data(line_powers_stage1_log)
    line_powers_s2_final = format_log_data(line_powers_stage2_log)

    # --- 构建最终返回的数据包 ---
    time_series_data = {
        "voltages_data_stage1": voltages_s1_final,
        "voltages_data_stage2": voltages_s2_final,
        "line_powers_data_stage1": line_powers_s1_final,
        "line_powers_data_stage2": line_powers_s2_final,
        "step_costs": step_costs_log,
        "pvw_pu": np.array(pvw_pu_log),
        "ess_soc_percent": np.array(ess_soc_log),
        "spot_kw": np.array(spot_kw_log),
        "sop_p_pu": np.array(sop_p_log),
        "sop_q_pu": np.array(sop_q_log),
        "nop_status": np.array(nop_status_log),
    }

    return metrics, time_series_data

def plot_sop_flows(all_ts_data, seed, gui_params):
    """
    为SOP的有功功率流(P)和无功功率流(Q)创建并列的对比图。
    """
    # 查找场景中SOP的数量和ID
    baseline_sops = all_ts_data.get('Baseline', {}).get('raw_baseline_data', {}).get('sop_flows', {})
    if not baseline_sops:
        print(f"场景(Seed: {seed}) 无SOP数据，跳过SOP潮流绘图。")
        return
    sop_ids = list(baseline_sops.keys())
    num_sops = len(sop_ids)

    # 创建时间轴
    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    # 检查是否有 'P1' 数据并获取步数
    if not sop_ids or 'P1' not in baseline_sops[sop_ids[0]] or not baseline_sops[sop_ids[0]]['P1']:
        print(f"场景(Seed: {seed}) SOP数据不完整，跳过SOP潮流绘图。")
        return
    num_steps = len(baseline_sops[sop_ids[0]]['P1'])
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    # 为每个SOP创建一行，每行包含P和Q两个子图
    fig, axes = plt.subplots(num_sops, 2, figsize=(18, 5 * num_sops), sharex=True, squeeze=False)
    fig.suptitle(f'各算法SOP有功/无功功率调度对比 (场景 Seed: {seed})', fontsize=18, fontproperties="SimHei")

    for i, sop_id in enumerate(sop_ids):
        ax_p = axes[i, 0]  # 左侧图用于绘制有功 P
        ax_q = axes[i, 1]  # 右侧图用于绘制无功 Q

        # --- 在左侧子图绘制有功功率 P ---
        ax_p.set_title(f'SOP ID: {sop_id} - 有功功率 (P)', fontproperties="SimHei")
        ax_p.set_ylabel('有功功率 P (pu)', fontproperties="SimHei")
        # 绘制Baseline的P曲线
        ax_p.plot(time_axis, baseline_sops[sop_id]['P1'], label=f'Baseline P1', color='blue', linestyle='-')
        # 绘制RL算法的P曲线
        for algo_name, ts_data in all_ts_data.items():
            if algo_name == 'Baseline' or 'sop_p_pu' not in ts_data:
                continue
            rl_sop_p_flows = ts_data['sop_p_pu']
            if rl_sop_p_flows.ndim == 2 and rl_sop_p_flows.shape[1] > i:
                ax_p.plot(time_axis, rl_sop_p_flows[:, i], label=f'{algo_name} P1', linestyle='--')
        ax_p.legend()
        ax_p.grid(True, linestyle=':')
        ax_p.axhline(0, color='black', linewidth=0.5)

        # --- 在右侧子图绘制无功功率 Q ---
        ax_q.set_title(f'SOP ID: {sop_id} - 无功功率 (Q)', fontproperties="SimHei")
        ax_q.set_ylabel('无功功率 Q (pu)', fontproperties="SimHei")
        # 绘制Baseline的Q曲线
        ax_q.plot(time_axis, baseline_sops[sop_id]['Q1'], label=f'Baseline Q1', color='green', linestyle='-')
        # 绘制RL算法的Q曲线
        for algo_name, ts_data in all_ts_data.items():
            if algo_name == 'Baseline' or 'sop_q_pu' not in ts_data:
                continue
            rl_sop_q_flows = ts_data['sop_q_pu']
            if rl_sop_q_flows.ndim == 2 and rl_sop_q_flows.shape[1] > i:
                ax_q.plot(time_axis, rl_sop_q_flows[:, i], label=f'{algo_name} Q1', linestyle='--')
        ax_q.legend()
        ax_q.grid(True, linestyle=':')
        ax_q.axhline(0, color='black', linewidth=0.5)

    # 统一设置X轴标签
    axes[-1, 0].set_xlabel('时间 (小时)', fontproperties="SimHei")
    axes[-1, 1].set_xlabel('时间 (小时)', fontproperties="SimHei")

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'sop_flows_PQ_seed_{seed}.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"【SOP增强版】场景(Seed: {seed})的SOP有功/无功对比图已保存至: {save_path}")


def plot_nop_status(all_ts_data, seed, gui_params):
    """为NOP的开关状态（0-断开, 1-闭合）创建对比图。"""
    baseline_nops = all_ts_data.get('Baseline', {}).get('raw_baseline_data', {}).get('nop_status', {})
    if not baseline_nops:
        print(f"场景(Seed: {seed}) 无NOP数据，跳过NOP状态绘图。")
        return
    nop_ids = list(baseline_nops.keys())
    num_nops = len(nop_ids)

    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    num_steps = len(baseline_nops[nop_ids[0]])
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    fig, axes = plt.subplots(num_nops, 1, figsize=(12, 3 * num_nops), sharex=True, squeeze=False)
    fig.suptitle(f'各算法NOP开关状态对比 (场景 Seed: {seed})', fontsize=16, fontproperties="SimHei")

    for i, nop_id in enumerate(nop_ids):
        ax = axes[i, 0]
        # 使用阶梯图(step plot)来绘制开关状态，更直观
        ax.step(time_axis, baseline_nops[nop_id], where='post', label='Baseline', color='blue')

        for algo_name, ts_data in all_ts_data.items():
            if algo_name == 'Baseline' or 'nop_status' not in ts_data:
                continue

            rl_nop_status = ts_data['nop_status']
            if rl_nop_status.shape[1] > i:
                ax.step(time_axis, rl_nop_status[:, i], where='post', label=algo_name, linestyle='--')

        ax.set_title(f'NOP ID: {nop_id}', fontproperties="SimHei")
        ax.set_ylabel('开关状态', fontproperties="SimHei")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['断开', '闭合'])
        ax.legend()
        ax.grid(True, axis='x', linestyle=':')

    axes[-1, 0].set_xlabel('时间 (小时)', fontproperties="SimHei")

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'nop_status_seed_{seed}.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"场景(Seed: {seed})的NOP状态对比图已保存至: {save_path}")


def print_baseline_status_monitor(data, grid, gui_params):
    """为Baseline的运行结果打印每一步的详细状态"""
    # 正常可以不需要这个函数，供debug使用
    print("\n" + "=" * 25 + " Baseline 运行状态监控 " + "=" * 25)

    # --- 初始化参数 ---
    sb_mva_kw = grid.SB * 1000
    step_seconds = gui_params['step_minutes'] * 60
    num_steps = len(data.get('grid_inflow_p', []))

    if num_steps == 0:
        print("无有效的时序数据可供监控。")
        print("=" * 75)
        return

    # --- 按时间步循环打印监控信息 ---
    for t in range(num_steps):
        print(f"\n---------- 步骤 {t} 状态监视 (Baseline) ----------")

        # --- 1. 需求侧 (无变化) ---
        base_load_kw = sum(bus.Pd(t * step_seconds) for bus in grid.Buses) * sb_mva_kw
        ev_charge_kw = sum(powers[t] * sb_mva_kw for powers in data.get('spot_powers', {}).values() if powers[t] > 0)
        ess_charge_kw = sum(powers[t] * sb_mva_kw for powers in data.get('ess_powers', {}).values() if powers[t] > 0)
        total_demand_kw = base_load_kw + ev_charge_kw + ess_charge_kw
        print(
            f"【需求侧】总需求: {total_demand_kw:.2f} kW (基础负荷: {base_load_kw:.2f}, EV充电: {ev_charge_kw:.2f}, ESS充电: {ess_charge_kw:.2f})")

        # --- 2. 电源侧 (此处的计算无变化，作为实际供给) ---
        pvw_devices = list(grid.PVWinds) if hasattr(grid, 'PVWinds') else []
        pvw_powers_data = data.get('pvw_powers', {})
        pv_gen_kw = sum(pvw_powers_data[dev.ID][t] * sb_mva_kw for dev in pvw_devices if
                        dev.Tag == 'pv' and dev.ID in pvw_powers_data)
        wind_gen_kw = sum(pvw_powers_data[dev.ID][t] * sb_mva_kw for dev in pvw_devices if
                          dev.Tag == 'wind' and dev.ID in pvw_powers_data)
        ess_discharge_kw = -sum(
            powers[t] * sb_mva_kw for powers in data.get('ess_powers', {}).values() if powers[t] < 0)
        ev_discharge_kw = -sum(
            powers[t] * sb_mva_kw for powers in data.get('spot_powers', {}).values() if powers[t] < 0)
        total_free_supply_kw = pv_gen_kw + wind_gen_kw + ess_discharge_kw + ev_discharge_kw

        grid_supply_kw = data['grid_inflow_p'][t] * sb_mva_kw
        gen_supply_kw = sum(powers[t] * sb_mva_kw for powers in data.get('gen_powers', {}).values())
        total_paid_supply_kw = grid_supply_kw + gen_supply_kw

        total_supply_kw = total_free_supply_kw + total_paid_supply_kw
        slack_supply_kw = sum(powers[t] * sb_mva_kw for powers in data.get('slack_powers', {}).values())

        print(f"【电源侧】(实际供给) 总计: {total_supply_kw + slack_supply_kw:.2f} kW")
        print(f"    ├─ 免费供给: {total_free_supply_kw:.2f} kW")
        print(f"    │   ├─ PV出力: {pv_gen_kw:.2f} kW")
        print(f"    │   ├─ Wind出力: {wind_gen_kw:.2f} kW")
        print(f"    │   ├─ ESS放电: {ess_discharge_kw:.2f} kW")
        print(f"    │   └─ V2G: {ev_discharge_kw:.2f} kW")
        print(
            f"    └─ 付费供给: {total_paid_supply_kw:.2f} kW (电网购买: {grid_supply_kw:.2f}, 本地发电: {gen_supply_kw:.2f})")
        if abs(slack_supply_kw) > 0.01:
            print(f"    ⚠️  警告: 系统存在功率缺口! Slack Power: {slack_supply_kw:.2f} kW")

        # --- 3. 网络调节设备 (无变化) ---
        sop_flows_data = data.get('sop_flows', {})
        nop_flows_data = data.get('nop_flows', {})
        total_sop_transfer_kw = sum(abs(flow['P1'][t]) * sb_mva_kw for flow in sop_flows_data.values())
        total_nop_transfer_kw = sum(abs(flow['P'][t]) * sb_mva_kw for flow in nop_flows_data.values())

        if total_sop_transfer_kw > 0.01 or total_nop_transfer_kw > 0.01:
            print(f"【网络调节】")
            if total_sop_transfer_kw > 0.01:
                sop_details = "; ".join(
                    [f"{sop_id}: {flow['P1'][t] * sb_mva_kw:.2f} kW" for sop_id, flow in sop_flows_data.items()])
                print(f"    ├─ SOP 传输有功: {sop_details}")
            if total_nop_transfer_kw > 0.01:
                nop_details = "; ".join(
                    [f"{nop_id}: {flow['P'][t] * sb_mva_kw:.2f} kW" for nop_id, flow in nop_flows_data.items()])
                print(f"    └─ NOP 传输有功: {nop_details}")

        # 可再生能源利用与弃用分析
        time_in_seconds = t * step_seconds

        # 计算理论最大出力
        total_potential_pv_kw = sum(dev.P(time_in_seconds) * sb_mva_kw for dev in pvw_devices if dev.Tag == 'pv')
        total_potential_wind_kw = sum(dev.P(time_in_seconds) * sb_mva_kw for dev in pvw_devices if dev.Tag == 'wind')

        # 计算弃光、弃风量 (理论最大 - 实际出力)
        curtailed_pv_kw = total_potential_pv_kw - pv_gen_kw
        curtailed_wind_kw = total_potential_wind_kw - wind_gen_kw

        # 仅当发生弃光或弃风时，才打印此部分
        if curtailed_pv_kw > 0.01 or curtailed_wind_kw > 0.01:
            print(f"【可再生能源利用】")
            if curtailed_pv_kw > 0.01:
                util_rate = (pv_gen_kw / total_potential_pv_kw * 100) if total_potential_pv_kw > 0 else 100
                print(f"    ├─ 光伏消纳: {pv_gen_kw:.2f} / {total_potential_pv_kw:.2f} kW (利用率: {util_rate:.1f}%)")
                print(f"    │   └─ 弃光功率: {curtailed_pv_kw:.2f} kW")
            if curtailed_wind_kw > 0.01:
                util_rate = (wind_gen_kw / total_potential_wind_kw * 100) if total_potential_wind_kw > 0 else 100
                print(
                    f"    └─ 风电消纳: {wind_gen_kw:.2f} / {total_potential_wind_kw:.2f} kW (利用率: {util_rate:.1f}%)")
                print(f"        └─ 弃风功率: {curtailed_wind_kw:.2f} kW")

        print("--------------------------------------------------")

    print("\n" + "=" * 75 + "\n")


def print_sop_monitor(data):
    """
    为Baseline的运行结果打印SOP的专项状态监视器。
    """
    print("\n" + "=" * 25 + " SOP 运行状态监控 " + "=" * 26)

    sop_flows = data.get('sop_flows')
    sop_slacks = data.get('sop_slacks')

    if not sop_flows:
        print("报告：模型结果中未找到SOP的运行数据。")
        print("=" * 75)
        return

    num_steps = 0
    # 通过任一SOP的数据确定总步数
    if sop_flows:
        first_sop_id = next(iter(sop_flows))
        num_steps = len(sop_flows[first_sop_id]['P1'])

    if num_steps == 0:
        print("报告：SOP数据为空，无法监控。")
        print("=" * 75)
        return

    print(f"{'Time':<6}{'SOP_ID':<8}{'P1(pu)':<12}{'Q1(pu)':<12}{'Loss(pu)':<12}{'Cap_Slack':<12}")
    print("-" * 75)

    for t in range(num_steps):
        for sop_id, flows in sop_flows.items():
            p1 = flows['P1'][t]
            q1 = flows['Q1'][t]
            loss = flows['Loss'][t]
            slack = sop_slacks.get(sop_id, [0] * num_steps)[t]

            # 只有当SOP有任何活动迹象时才打印，避免刷屏
            if abs(p1) > 1e-4 or abs(q1) > 1e-4 or abs(loss) > 1e-4 or abs(slack) > 1e-4:
                print(f"{t:<6}{sop_id:<8}{p1:<12.4f}{q1:<12.4f}{loss:<12.4f}{slack:<12.4f}")

    print("=" * 75 + "\n")


def evaluate_baseline(gui_params, seed, stations_list, grid, use_two_stage=False):
    """
    - 在 two_stage 模式下，临时向松弛节点添加一个虚拟发电机，用于精确潮流计算。
    - 确保在任何情况下都返回与RL Agent完全兼容的数据结构。
    """
    VIRTUAL_GEN_ID = 'gen_for_slack_bus'
    slack_bus_id = gui_params.get('slack_bus', 'b1')

    # 临时添加的虚拟发电机对象
    slack_generator = None

    # --- 阶段 1: 总是先运行 Linear DistFlow 优化 ---
    print("\n--- Running Baseline Stage 1 (Linear DistFlow Optimization) ---")
    result, stage_one_data, _ = solve_baseline(grid, stations_list, gui_params)

    if not stage_one_data:
        print("!!!!!! Baseline Stage 1 求解失败, 无法继续 !!!!!!")
        metrics = {k: np.nan for k in
                    ["总成本", "购电成本", "发电成本", "SOP损耗成本", "ESS放电成本", "电压合格率(%)",
                    "EV充电满足率(%)"]}
        return metrics, {}

    if use_two_stage:
        metrics, time_series_data = run_baseline_stage_two(grid, stage_one_data, gui_params)

        #从第一阶段的结果中提取数据，以保持与RL Agent和单阶段模式的数据结构兼容，从而让绘图函数能够正常工作。
        time_series_data["pvw_pu"] = np.array(list(stage_one_data.get('pvw_powers', {}).values())).T
        time_series_data["ess_soc_percent"] = np.array(list(stage_one_data.get('ess_soc', {}).values())).T * 100
        time_series_data["spot_kw"] = np.array(list(stage_one_data.get('spot_powers', {}).values())).T * (
                    grid.SB * 1000)

        # 打印第二阶段的购电总量2. EV的不支持电池老化模型，也不支持基于用户意愿的V2G调度逻辑，只针对于电网的成本最低来调用V2G。
        total_stage2_inflow_kwh = (metrics.get('购电成本', 0) / load_electricity_price(gui_params=gui_params)[
            0]) if load_electricity_price(gui_params=gui_params) else 0
        # 更精确的计算方式
        if time_series_data and 'raw_baseline_data' in time_series_data:
            total_stage2_inflow_kwh = sum(
                p * (gui_params['step_minutes'] / 60.0) * grid.SB * 1000
                for p in time_series_data['raw_baseline_data'].get('grid_inflow_p_stage2', [])  # 假设stage2结果存在这里
            ) if 'grid_inflow_p_stage2' in time_series_data['raw_baseline_data'] else metrics.get('购电成本', 0) / (
                        sum(load_electricity_price(gui_params=gui_params)) / len(
                    load_electricity_price(gui_params=gui_params)))



        return metrics, time_series_data
    else:
        # 单阶段模式的逻辑保持不变
        print("--- Baseline 评估模式: Single-Stage, 正在构建完整数据包 ---")
        metrics = {
                "购电成本": stage_one_data.get('grid_purchase_cost', 0),
                "发电成本": stage_one_data.get('generation_cost', 0),
                "SOP损耗成本": stage_one_data.get('sop_loss_cost', 0),
                "ESS放电成本": stage_one_data.get('ess_discharge_cost', 0),
        }
        metrics["总成本"] = sum(metrics.values())
        #将包含所有惩罚的完整目标函数值也加入到metrics中
        metrics["总目标值"] = stage_one_data.get('objective_value', metrics["总成本"])
        violations = 0
        total_checks = sum(len(v) for v in stage_one_data.get('bus_voltages', {}).values())
        if total_checks > 0:
            for voltages in stage_one_data.get('bus_voltages', {}).values():
                standards = EVALUATION_CONFIG["standards"]
                for v in voltages:
                    if not (standards["voltage_min_pu"] <= v <= standards["voltage_max_pu"]):
                        violations += 1
            metrics["电压合格率(%)"] = (1 - violations / total_checks) * 100
        else:
            metrics["电压合格率(%)"] = 100.0

        satisfied_count = stage_one_data.get('charged_ev_count', 0)
        total_count = stage_one_data.get('total_ev_count', 0)
        metrics["EV充电满足率(%)"] = (satisfied_count / total_count) * 100 if total_count > 0 else 100

        num_steps = len(stage_one_data.get('grid_inflow_p', []))
        step_costs = [0] * num_steps
        if num_steps > 0:
            price = load_electricity_price(gui_params=gui_params)
            sb_mva = grid.SB
            step_h = gui_params['step_minutes'] / 60.0
            for t in range(num_steps):
                grid_cost_t = price[t] * stage_one_data.get('grid_inflow_p', [0] * num_steps)[
                    t] * sb_mva * 1000 * step_h
                gen_cost_t = sum(
                    (gen.CostA(t) * (p_list[t] * sb_mva * 1000) ** 2 + gen.CostB(t) * (
                            p_list[t] * sb_mva * 1000) + gen.CostC(t)) * step_h
                    for gen_id, p_list in stage_one_data.get('gen_powers', {}).items() if (gen := grid.Gen(gen_id))
                )
                sop_loss_cost_t = sum(
                    price[t] * flow_data['Loss'][t] * sb_mva * 1000 * step_h
                    for sop_id, flow_data in stage_one_data.get('sop_flows', {}).items()
                )
                step_costs[t] = grid_cost_t + gen_cost_t + sop_loss_cost_t

        voltages_data = stage_one_data.get("bus_voltages", {})

        line_powers_data = stage_one_data.get("line_powers", {})
        time_series_data = {
                "step_costs": step_costs,
                "voltages_data_stage1": voltages_data,
                "voltages_data_stage2": voltages_data,
                # 将潮流数据复制到两个新的键中，以匹配绘图函数的期望
                "line_powers_data_stage1": line_powers_data,
                "line_powers_data_stage2": line_powers_data,
                "raw_baseline_data": stage_one_data,
                "pvw_pu": np.array(list(stage_one_data.get('pvw_powers', {}).values())).T,
                "ess_soc_percent": np.array(list(stage_one_data.get('ess_soc', {}).values())).T * 100,
                "spot_kw": np.array(list(stage_one_data.get('spot_powers', {}).values())).T * (
                        grid.SB * 1000)
        }
        return metrics, time_series_data
def plot_accumulated_costs(all_metrics, all_ts_data, seed, gui_params, grid_for_params):
    """
    绘制所有算法的累计成本对比图。
    """
    plt.figure(figsize=(12, 8))

    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']

    # 预加载成本计算所需参数
    price = load_electricity_price(gui_params=gui_params)
    sb_mva = grid_for_params.SB
    step_h = gui_params['step_minutes'] / 60.0
    gen_costs_A = {g.ID: g.CostA(0) for g in grid_for_params.Gens}
    gen_costs_B = {g.ID: g.CostB(0) for g in grid_for_params.Gens}
    gen_costs_C = {g.ID: g.CostC(0) for g in grid_for_params.Gens}

    for algo_name, metrics in all_metrics.items():
        ts_data = all_ts_data.get(algo_name)
        if not ts_data or 'step_costs' not in ts_data:
            print(f"警告：算法 {algo_name} 缺少 'step_costs' 数据，无法绘制累计成本。")
            continue

        step_costs = ts_data['step_costs']
        if len(step_costs) == 0:
            continue

        # 计算累计成本
        accumulated_costs = np.cumsum(step_costs)
        time_axis = np.linspace(start_hour, end_hour, len(accumulated_costs))

        # 绘制曲线
        plt.plot(time_axis, accumulated_costs, label=f"{algo_name} (总成本: {metrics.get('总成本', 0):.2f} 元)",
                 marker='o', markersize=3, linestyle='-')

    plt.title(f'各算法累计运行成本对比 (场景 Seed: {seed})', fontproperties="SimHei", size=16)
    plt.xlabel('时间 (小时)', fontproperties="SimHei")
    plt.ylabel('累计成本 (元)', fontproperties="SimHei")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'accumulated_cost_seed_{seed}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"场景(Seed: {seed})的累计成本对比图已保存至: {save_path}")

def plot_aggregated_ev_power(all_ts_data, seed, gui_params):
    """
    绘制包含6条独立曲线的EV总功率图。
    - 先分离每个桩的充/放电功率，然后再分别求和，避免功率对冲。
    """
    # 检查是否有有效数据
    algorithms = list(all_ts_data.keys())
    if not algorithms or not all_ts_data[algorithms[0]]:
        print(f"场景(Seed: {seed}) 无有效的充电桩时序数据，跳过总功率绘图。")
        return

    # 创建时间轴
    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    try:
        num_steps = len(next(iter(all_ts_data.values()))['spot_kw'])
        if num_steps == 0: raise IndexError
    except (StopIteration, KeyError, IndexError):
        print(f"数据步数为零，跳过场景(Seed: {seed})的总功率绘图。")
        return
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    # 开始绘图
    fig, ax = plt.subplots(figsize=(15, 8))

    # —— 颜色映射+ 自动分配 ——
    algorithms = list(all_ts_data.keys())

    def _normalize_algo_name(name: str) -> str:
        s = name.strip()
        s = re.sub(r'(?i)_(two|single)_?stage', '', s)
        s = re.sub(r'(?i)\b(two|single)\s*stage\b', '', s)
        s = re.sub(r'[_\-\s]+seed\d+', '', s)
        s = re.sub(r'[_\-\s]+v\d+', '', s)
        s = re.sub(r'[_\-\s]+', '_', s).upper()
        s = s.replace('STABLE_BASELINES3_', '')
        return s

    BASE_COLOR_MAP = {
        'BASELINE': 'blue',
        'DDPG': '#e41a1c',
        'PPO': '#377eb8',
        'SAC': '#4daf4a',
        'TD3': '#984ea3',
        'A2C': '#ff7f00',
        'TRPO': '#a65628',
        'DQN': '#f781bf',
    }

    FALLBACK_PALETTE = [
        '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
        '#e6ab02', '#a6761d', '#666666', '#a6cee3', '#fb9a99'
    ]

    unknown_keys = sorted({
        _normalize_algo_name(a)
        for a in algorithms
        if _normalize_algo_name(a) not in BASE_COLOR_MAP and _normalize_algo_name(a) != 'BASELINE'
    })

    algo_colors = {}
    for a in algorithms:
        key = _normalize_algo_name(a)
        if key == 'BASELINE':
            algo_colors[a] = BASE_COLOR_MAP['BASELINE']
            continue
        color = BASE_COLOR_MAP.get(key)
        if color is None:
            idx = list(unknown_keys).index(key) % len(FALLBACK_PALETTE)
            color = FALLBACK_PALETTE[idx]
        algo_colors[a] = color

    for algo_name, ts_data in all_ts_data.items():
        if 'spot_kw' not in ts_data or ts_data['spot_kw'].size == 0:
            continue

        # spot_powers 的维度是 (时间步数, 充电桩数量)
        spot_powers = ts_data['spot_kw']

        # 1. 先在每个充电桩的层面上，分离出充电功率（正值）和放电功率（负值）
        all_charge_powers = np.maximum(spot_powers, 0)
        all_discharge_powers = np.minimum(spot_powers, 0)

        # 2. 然后，再沿着充电桩的维度（axis=1）分别求和
        total_charge_curve = np.sum(all_charge_powers, axis=1)
        total_discharge_curve = np.sum(all_discharge_powers, axis=1)

        # 获取该算法的基础颜色
        color = algo_colors.get(algo_name, 'gray')

        # 绘制两条独立的曲线：一条充电，一条放电
        # 用【实线】绘制充电功率曲线
        ax.plot(time_axis, total_charge_curve, color=color, linestyle='-', linewidth=2, label=f'{algo_name} 充电功率')

        # 用【虚线】绘制放电功率曲线
        ax.plot(time_axis, total_discharge_curve, color=color, linestyle='--', linewidth=2, label=f'{algo_name} 放电功率')

    # --- 图表美化 ---
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_title(f'各算法下EV充/放电总功率精细对比 (场景 Seed: {seed})', fontproperties="SimHei", size=16)
    ax.set_xlabel('时间 (小时)', fontproperties="SimHei")
    ax.set_ylabel('总功率 (kW)', fontproperties="SimHei")
    ax.legend(fontsize=12, ncol=3)
    ax.grid(True, linestyle=':', alpha=0.6)

    # --- 保存图表 ---
    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'aggregated_ev_power_corrected_{seed}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"【逻辑已修正】场景(Seed: {seed})的EV充放电分离对比图已保存至: {save_path}")
def plot_and_save_results(all_ts_data, seed, gui_params):
    """将所有算法的时序数据绘制对比图并保存到Excel"""

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, f'comparison_data_seed_{seed}.xlsx')

    algorithms = list(all_ts_data.keys())
    if not algorithms or not all_ts_data[algorithms[0]]:
        print(f"场景(Seed: {seed}) 无有效的时序数据，跳过绘图和保存。")
        return

    # 创建时间轴
    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    try:
        # 以第一个有效数据为基准确定步数
        num_steps = len(next(iter(all_ts_data.values()))['pvw_pu'])
        if num_steps == 0: raise IndexError
    except (StopIteration, KeyError, IndexError):
        print(f"数据步数为零，跳过场景(Seed: {seed})的绘图和保存。")
        return

    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    # --- 开始绘图 ---
    # 1. 光伏和风电总出力
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        total_pvw_output = np.sum(all_ts_data[algo]['pvw_pu'], axis=1)
        plt.plot(time_axis, total_pvw_output, label=algo, marker='o', linestyle='--')
    plt.title(f'光伏与风电总出力对比 (场景 Seed: {seed})', fontproperties="SimHei", size=16)
    plt.xlabel('时间 (小时)', fontproperties="SimHei")
    plt.ylabel('总出力 (pu)', fontproperties="SimHei")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'pvw_total_output_seed_{seed}.png'))
    plt.close()

    # 2. ESS SOC变化
    num_ess = next(iter(all_ts_data.values())).get('ess_soc_percent', np.array([])).shape[1]
    if num_ess > 0:
        fig, axes = plt.subplots(num_ess, 1, figsize=(12, 4 * num_ess), sharex=True, squeeze=False)
        fig.suptitle(f'储能系统(ESS) SOC变化对比 (场景 Seed: {seed})', fontsize=16, fontproperties="SimHei")
        for i in range(num_ess):
            for algo in algorithms:
                soc_data_to_plot = all_ts_data[algo]['ess_soc_percent'][:-1, i]
                axes[i, 0].plot(time_axis, soc_data_to_plot, label=algo, marker='.')
            axes[i, 0].set_title(f'ESS #{i + 1}', fontproperties="SimHei")
            axes[i, 0].set_ylabel('SOC (%)', fontproperties="SimHei")
            axes[i, 0].legend()
            axes[i, 0].grid(True)
        axes[-1, 0].set_xlabel('时间 (小时)', fontproperties="SimHei")
        plt.savefig(os.path.join(output_dir, f'ess_soc_seed_{seed}.png'))
        plt.close()

    # 3. 各充电桩充电功率
    num_spots = next(iter(all_ts_data.values())).get('spot_kw', np.array([])).shape[1]
    if num_spots > 0:
        fig, axes = plt.subplots(num_spots, 1, figsize=(12, 3 * num_spots), sharex=True, squeeze=False)
        fig.suptitle(f'充电桩功率变化对比 (场景 Seed: {seed})', fontsize=16, fontproperties="SimHei")
        for i in range(num_spots):
            for algo in algorithms:
                axes[i, 0].plot(time_axis, all_ts_data[algo]['spot_kw'][:, i], label=algo, alpha=0.8)
            axes[i, 0].set_title(f'充电桩 #{i + 1}', fontproperties="SimHei")
            axes[i, 0].set_ylabel('功率 (kW)', fontproperties="SimHei")
            axes[i, 0].legend()
            axes[i, 0].grid(True)
            axes[i, 0].axhline(0, color='black', linewidth=0.5)
        axes[-1, 0].set_xlabel('时间 (小时)', fontproperties="SimHei")
        plt.savefig(os.path.join(output_dir, f'spot_power_seed_{seed}.png'))
        plt.close()

    print(f"场景(Seed: {seed})的对比图已保存至文件夹: {output_dir}")

    # --- 开始保存数据到Excel ---
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 工作表1: 光伏和风电总出力
        pvw_df_data = {'Time(h)': time_axis}
        for algo in algorithms:
            pvw_pu_data = all_ts_data[algo].get('pvw_pu', np.full((num_steps, 0), np.nan))
            pvw_df_data[f'{algo}_Total_PVW_Output(pu)'] = np.sum(pvw_pu_data,
                                                                 axis=1) if pvw_pu_data.ndim > 1 else pvw_pu_data
        pd.DataFrame(pvw_df_data).to_excel(writer, sheet_name='PV_Wind_Total_Output', index=False)

        # 工作表2: ESS SOC
        if num_ess > 0:
            ess_df_data = {'Time(h)': time_axis}
            for i in range(num_ess):
                for algo in algorithms:
                    ess_soc_data = all_ts_data[algo].get('ess_soc_percent', np.full((num_steps, num_ess), np.nan))
                    soc_data_to_save = ess_soc_data[:-1, i]
                    ess_df_data[f'{algo}_ESS_{i + 1}_SOC(%)'] = soc_data_to_save
            pd.DataFrame(ess_df_data).to_excel(writer, sheet_name='ESS_SOC', index=False)

        # 工作表3: EV 充电桩功率
        if num_spots > 0:
            spot_df_data = {'Time(h)': time_axis}
            for i in range(num_spots):
                for algo in algorithms:
                    spot_kw_data = all_ts_data[algo].get('spot_kw', np.full((num_steps, num_spots), np.nan))
                    spot_df_data[f'{algo}_Spot_{i + 1}_Power(kW)'] = spot_kw_data[:, i]
            pd.DataFrame(spot_df_data).to_excel(writer, sheet_name='EV_Spot_Powers', index=False)

            #保存所有节点的电压数据
            # 【母线电压】
            print("...正在写入'Bus_Voltages'工作表")
            all_buses = set()
            for data in all_ts_data.values():
                voltages_data_dict = data.get('voltages_data_stage2', {})
                # 检查它是否为字典 (dict)，并从字典的键中更新母线列表
                if voltages_data_dict and isinstance(voltages_data_dict, dict):
                    all_buses.update(voltages_data_dict.keys())

            if all_buses:
                def get_sort_key(bus_id):
                    try:
                        # 尝试按数字部分排序 (例如 'b10' 排在 'b2' 之后)
                        return int(''.join(filter(str.isdigit, bus_id)))
                    except:
                        return bus_id  # 如果没有数字，按原字符串排序

                sorted_buses = sorted(list(all_buses), key=get_sort_key)
                voltage_df_data = {'Time_Step': list(range(num_steps))}

                for bus_id in sorted_buses:
                    for algo_name, ts_data in all_ts_data.items():
                        col_name = f"{algo_name}_{bus_id}_V(pu)"
                        voltages_data_dict = ts_data.get('voltages_data_stage2', {})
                        # 数据现在是 dict[list]，直接用 bus_id 作为键来获取电压列表
                        voltage_list = voltages_data_dict.get(bus_id, [np.nan] * num_steps)
                        # 确保列表长度与时间步一致
                        voltage_df_data[col_name] = voltage_list[:num_steps]

                pd.DataFrame(voltage_df_data).to_excel(writer, sheet_name='Bus_Voltages', index=False)

            # 【 线路潮流】
            print("...正在写入'Line_Flows'工作表")
            all_lines = set()
            for data in all_ts_data.values():
                flows_data_dict = data.get('line_powers_data_stage2', {})
                if flows_data_dict and isinstance(flows_data_dict, dict):
                    all_lines.update(flows_data_dict.keys())

            if all_lines:
                def get_line_sort_key(line_id):
                    try:
                        return int(''.join(filter(str.isdigit, line_id)))
                    except:
                        return line_id

                sorted_lines = sorted(list(all_lines), key=get_line_sort_key)
                flow_df_data = {'Time_Step': list(range(num_steps))}

                for line_id in sorted_lines:
                    for algo_name, ts_data in all_ts_data.items():
                        col_name = f"{algo_name}_{line_id}_P(pu)"
                        flow_data_dict = ts_data.get('line_powers_data_stage2', {})
                        flow_list = flow_data_dict.get(line_id, [np.nan] * num_steps)
                        flow_df_data[col_name] = flow_list[:num_steps]

                pd.DataFrame(flow_df_data).to_excel(writer, sheet_name='Line_Flows', index=False)
            print("...正在写入'SOP_Flows'工作表")
            baseline_sops = all_ts_data.get('Baseline', {}).get('raw_baseline_data', {}).get('sop_flows', {})
            if baseline_sops:
                sop_ids = list(baseline_sops.keys())
                sop_df_data = {'Time_Step': list(range(num_steps))}
                for sop_id in sop_ids:
                    # 添加Baseline数据
                    sop_df_data[f"Baseline_{sop_id}_P(pu)"] = baseline_sops[sop_id]['P1']
                    sop_df_data[f"Baseline_{sop_id}_Q(pu)"] = baseline_sops[sop_id]['Q1']

                    # 添加RL数据
                    for algo_name, ts_data in all_ts_data.items():
                        if algo_name != 'Baseline':
                            sop_p_data = ts_data.get('sop_p_pu', np.array([]))
                            sop_q_data = ts_data.get('sop_q_pu', np.array([]))
                            sop_idx = sop_ids.index(sop_id)
                            if sop_p_data.shape[1] > sop_idx:
                                sop_df_data[f"{algo_name}_{sop_id}_P(pu)"] = sop_p_data[:, sop_idx]
                                sop_df_data[f"{algo_name}_{sop_id}_Q(pu)"] = sop_q_data[:, sop_idx]
                pd.DataFrame(sop_df_data).to_excel(writer, sheet_name='SOP_Flows', index=False)

            # NOP数据保存
            print("...正在写入'NOP_Status'工作表")
            baseline_nops = all_ts_data.get('Baseline', {}).get('raw_baseline_data', {}).get('nop_status', {})
            if baseline_nops:
                nop_ids = list(baseline_nops.keys())
                nop_df_data = {'Time_Step': list(range(num_steps))}
                for nop_id in nop_ids:
                    # 添加Baseline数据
                    nop_df_data[f"Baseline_{nop_id}_Status"] = baseline_nops[nop_id]

                    # 添加RL数据
                    for algo_name, ts_data in all_ts_data.items():
                        if algo_name != 'Baseline':
                            nop_status_data = ts_data.get('nop_status', np.array([]))
                            nop_idx = nop_ids.index(nop_id)
                            if nop_status_data.shape[1] > nop_idx:
                                nop_df_data[f"{algo_name}_{nop_id}_Status"] = nop_status_data[:, nop_idx]
                pd.DataFrame(nop_df_data).to_excel(writer, sheet_name='NOP_Status', index=False)

    print(f"场景(Seed: {seed})的详细时序数据已保存至Excel文件: {os.path.abspath(excel_path)}")


def plot_ev_occupancy(env_instance, seed, gui_params):
    """
    绘制特定场景下，充电站内实时在场车辆数量的变化曲线。
    这是一个诊断工具，用于验证充电功率是否与车辆数量匹配。
    """
    if not hasattr(env_instance, 'ev_present'):
        print("诊断错误：无法从环境中找到 ev_present 数据。")
        return

    # 从环境中获取 ev_present 数据 (这是一个 桩数 x 时间步 的 0/1 矩阵)
    ev_present_matrix = env_instance.ev_present

    # 沿着充电桩的维度（axis=0）求和，得到每个时间步的在场车辆总数
    num_cars_present_per_step = np.sum(ev_present_matrix, axis=0)

    # 创建时间轴
    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    num_steps = len(num_cars_present_per_step)
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    # 绘图
    plt.figure(figsize=(15, 6))
    plt.plot(time_axis, num_cars_present_per_step, label='实时在场车辆数', color='purple', drawstyle='steps-post')

    plt.title(f'充电站实时在场车辆数变化 (场景 Seed: {seed})', fontproperties="SimHei", size=16)
    plt.xlabel('时间 (小时)', fontproperties="SimHei")
    plt.ylabel('在场车辆数 (辆)', fontproperties="SimHei")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()

    # 保存图表
    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'ev_occupancy_seed_{seed}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"【诊断工具】场景(Seed: {seed})的在场车辆数曲线图已保存至: {save_path}")


if __name__ == '__main__':
    # =================================================================================
    # 1. 从 config.py 加载核心配置
    # =================================================================================
    EVALUATION_MODE = EVALUATION_CONFIG["flow_mode"]
    gui_params = CORE_PARAMS
    num_test_episodes = EVALUATION_CONFIG["num_test_episodes"]

    print(f"\n{'=' * 35}")
    print(f"  当前评估模式: {EVALUATION_MODE.upper()}")
    print(f"{'=' * 35}\n")

    # =================================================================================
    # 2. 动态发现和加载所有可用的算法类
    #    - 首先加载stable_baselines3的内置算法作为基础。
    #    - 然后扫描 custom_algorithms/ 文件夹，加载所有用户自定义的算法插件。
    # =================================================================================
    print(f"{'=' * 35}")
    print("  正在动态发现和加载所有可用算法...")


    # 定义一个函数来执行发现和加载的逻辑，以保持主流程清晰
    def discover_and_load_algorithms():
        model_class_registry = {}

        # 2.1. 加载 stable-baselines3 的内置算法
        from stable_baselines3 import PPO, DDPG, SAC, TD3
        model_class_registry.update({"PPO": PPO, "DDPG": DDPG, "SAC": SAC, "TD3": TD3})

        # 2.2. 扫描 custom_algorithms 目录下的所有 python 文件
        plugin_dir = "custom_algorithms"
        if not os.path.isdir(plugin_dir):
            print(f"  信息: 未找到插件目录 'izer{plugin_dir}'，将仅使用内置算法。")
            return model_class_registry

        import glob
        import importlib.util
        plugin_files = glob.glob(os.path.join(plugin_dir, "*.py"))

        for plugin_file in plugin_files:
            module_name = os.path.basename(plugin_file)[:-3]
            try:
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, 'register_algorithm'):
                    registration_info = module.register_algorithm()
                    algo_name = registration_info['name']
                    algo_class = registration_info['class']
                    model_class_registry[algo_name] = algo_class
                    print(f"  [插件加载器] 成功加载自定义算法: '{algo_name}'")
                else:
                    print(
                        f"  [插件加载器] 警告: 文件 {os.path.basename(plugin_file)} 不是有效的插件 (缺少 register_algorithm 函数)。")
            except Exception as e:
                print(f"  [插件加载器] 错误: 加载插件 {os.path.basename(plugin_file)} 失败: {e}")

        return model_class_registry


    # 执行发现操作
    model_name_to_class = discover_and_load_algorithms()
    print(f"  加载完成！当前可用算法: {list(model_name_to_class.keys())}")
    print(f"{'=' * 35}\n")

    # =================================================================================
    # 3. 【自动化核心】自动发现在 models/ 目录中所有训练好的模型
    #    - 不再读取 config.py 中的硬编码列表，而是直接扫描文件夹。
    # =================================================================================
    print(f"{'=' * 35}")
    print("  正在自动扫描 'models/' 目录以发现已训练的模型...")
    models_to_evaluate = {}
    model_folders = [f for f in os.listdir(PATHS["models_dir"]) if os.path.isdir(os.path.join(PATHS["models_dir"], f))]

    for folder_name in model_folders:
        # e.g., folder_name = "best_brilliantalgo_two_stage" -> display_name = "BrilliantAlgo_Two_Stage"
        display_name = folder_name.replace("best_", "").replace("_", " ").title().replace(" ", "_")
        model_path = os.path.join(PATHS["models_dir"], folder_name, "best_model.zip")

        if not os.path.exists(model_path):
            print(f"  警告: 在文件夹 '{folder_name}' 中未找到 'best_model.zip'，跳过。")
            continue

        # 从模型名称推断其类型，并在已注册的算法中查找对应的类
        model_type_key = next((key for key in model_name_to_class if key in display_name.upper()), None)

        if model_type_key:
            model_class = model_name_to_class[model_type_key]
            models_to_evaluate[display_name] = (model_path, model_class)
            print(f"  [模型发现器] 发现模型 '{display_name}' 并关联到 '{model_type_key}' 类。")
        else:
            print(f"  警告: 在已注册的算法中找不到与 '{display_name}' 匹配的类，跳过此模型。")
    print(f"  发现完成！将评估以下模型: {list(models_to_evaluate.keys())}")
    print(f"{'=' * 35}\n")

    # =================================================================================
    # 4. 初始化环境和结果记录器
    # =================================================================================
    all_results_metrics = []
    try:
        use_two_stage = (EVALUATION_MODE == 'two_stage')
        rl_env = PowerGridEnv(gui_params=gui_params, use_two_stage_flow=use_two_stage)
    except Exception as e:
        print(f"初始化PowerGridEnv失败: {e}")
        traceback.print_exc()
        exit()

    # =================================================================================
    # 5. 主评估循环
    #    - 遍历所有测试场景 (episodes)。
    #    - 在每个场景中，依次运行 Baseline 和所有已发现的RL模型。
    #    - 为每个场景生成图表和数据文件。
    # =================================================================================
    for i in range(num_test_episodes):
        seed = i
        print(f"\n--- 开始评估场景 {i + 1}/{num_test_episodes} (seed={seed}) ---\n")

        all_metrics_for_this_seed = {}
        time_series_log_for_this_seed = {}

        # 核心同步逻辑：先用带seed的RL环境生成标准场景，供Baseline和RL Agent共同使用
        rl_env.reset(seed=seed)
        stations_list_for_this_seed = rl_env.stations_list
        grid_instance_for_this_seed = rl_env.grid

        # 5.1. 评估 Baseline
        try:
            grid_for_baseline = deepcopy(grid_instance_for_this_seed)
            baseline_metrics, baseline_ts = evaluate_baseline(gui_params, seed, stations_list_for_this_seed,
                                                              grid_for_baseline,
                                                              use_two_stage=use_two_stage)
            if baseline_metrics and not pd.isna(baseline_metrics.get("总成本")):
                baseline_metrics['算法'] = 'Baseline'
                all_results_metrics.append(baseline_metrics)
                all_metrics_for_this_seed['Baseline'] = baseline_metrics
                time_series_log_for_this_seed['Baseline'] = baseline_ts
        except Exception as e:
            print(f"评估 Baseline 时发生严重错误: {e}")
            traceback.print_exc()

        # 5.2. 评估所有已发现的 RL 模型
        for model_name, (model_path, model_class) in models_to_evaluate.items():
            try:
                model = model_class.load(model_path, env=rl_env)
                rl_metrics, rl_ts = evaluate_rl_agent(model, rl_env, seed)
                rl_metrics['算法'] = model_name
                all_results_metrics.append(rl_metrics)
                all_metrics_for_this_seed[model_name] = rl_metrics
                time_series_log_for_this_seed[model_name] = rl_ts
            except Exception as e:
                print(f"评估 {model_name} 时发生严重错误: {e}")
                traceback.print_exc()

        # 5.3. 为当前场景生成所有对比图和数据文件
        if time_series_log_for_this_seed:
            print("\n--- 正在为当前场景生成可视化报告和数据文件... ---")
            # (调用所有绘图函数)
            plot_and_save_results(time_series_log_for_this_seed, seed, gui_params)
            plot_accumulated_costs(all_metrics_for_this_seed, time_series_log_for_this_seed, seed, gui_params,
                                   grid_instance_for_this_seed)
            plot_voltage_snapshots(time_series_log_for_this_seed, seed, gui_params)
            plot_line_flow_snapshots_comparison(time_series_log_for_this_seed, seed, gui_params)
            plot_aggregated_ev_power(time_series_log_for_this_seed, seed, gui_params)
            plot_sop_flows(time_series_log_for_this_seed, seed, gui_params)
            plot_nop_status(time_series_log_for_this_seed, seed, gui_params)
            print("--- 报告与数据文件生成完毕。 ---\n")
        else:
            print(f"场景(Seed: {seed}) 无有效的评估结果，跳过绘图。")

    # =================================================================================
    # 6. 最终结果汇总
    #    - 打印所有场景的平均性能指标对比表。
    # =================================================================================
    if not all_results_metrics:
        print("\n所有评估均未成功，无法生成汇总报告。")
    else:
        results_df = pd.DataFrame(all_results_metrics)
        display_columns = [
            "总成本", "购电成本", "发电成本", "SOP损耗成本", "ESS放电成本",
            "精确总网损(kW)", "电压合格率(%)", "EV充电满足率(%)"
        ]
        existing_display_columns = [col for col in display_columns if col in results_df.columns]
        summary = results_df.groupby('算法')[existing_display_columns].mean()
        print("\n\n" + "=" * 110)
        print(" " * 45 + "仿 真 评 估 结 果 汇 总" + " " * 45)
        print("=" * 110)
        print(summary.to_string(float_format="%.4f"))
        print("=" * 110)
