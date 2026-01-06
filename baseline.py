"""
该模块定义了平台的"基准"（Baseline）算法。
核心功能是使用Pyomo构建一个全局最优潮流模型（基于线性化的DistFlow），
一次性计算出在整个仿真周期（例如24小时）内，所有可控设备（发电机、储能、
充电桩、SOP/NOP等）的最优调度策略，以实现总运行成本的最小化。
这个计算结果被用作评估强化学习智能体性能的"最优标杆"。
"""

import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
from fpowerkit import Grid
from fpowerkit.solbase import GridSolveResult
from config import CORE_PARAMS, BASELINE_PARAMS, EVALUATION_CONFIG
from grid_model import load_station_info  # 导入以获取电动汽车充电站母线ID
from grid_model import load_electricity_price  # 导入以获取电价数据
from pyomo.environ import Binary, NonNegativeReals
from pyomo.opt import SolverStatus, TerminationCondition
from gev_station import GEVStation
from pyomo.environ import value  # 确保文件顶部有此导入
def create_baseline_model(grid: Grid, stations_list, time_steps: int, gui_params: dict): # <--- 接收 station 对象

    """
    使用 Pyomo 创建并求解24小时完整的线性化 DistFlow 最优潮流模型。
    控制充电桩功率分配，假设所有充电桩、分布式能源、SOP 和 NOP 可控，计算全局最优解。
    """
    # 初始化 Pyomo 模型
    model = ConcreteModel(name="LinearDistFlowBaseline")
    sb_mva = grid.SB
    # 1. 在函数最开始，就从 grid 对象中获取所有组件列表。
    buses = list(grid.Buses)
    lines = list(grid.ActiveLines)
    pvws = list(grid.PVWinds)
    esss = list(grid.ESSs)
    sops = list(getattr(grid, 'SOPs', {}).values())
    nops = list(getattr(grid, 'NOPs', {}).values())

    # 2. 对发电机列表进行一次性的、决定性的过滤。
    #    这个增强的过滤逻辑会移除所有为第二阶段准备的虚拟/等效发电机。
    print("--- [BASELINE FIX] Filtering generators for Stage 1 optimization... ---")
    VIRTUAL_GEN_ID = 'gen_for_slack_bus'
    gens_to_optimize = [
        g for g in grid.Gens
        if g.ID != VIRTUAL_GEN_ID and 'gen_for_sop' not in g.ID
    ]
    print(f"  - Original gen count: {len(list(grid.Gens))}")
    print(f"  - Gens after filtering: {len(gens_to_optimize)}")

    # 后续所有操作都基于这个干净的 `gens_to_optimize` 列表
    gens = gens_to_optimize

    bus_dict = {bus.ID: idx for idx, bus in enumerate(buses)}
    line_dict = {line.ID: idx for idx, line in enumerate(lines)}
    gen_dict = {gen.ID: idx for idx, gen in enumerate(gens)}
    pvw_dict = {pvw.ID: idx for idx, pvw in enumerate(pvws)}
    ess_dict = {ess.ID: idx for idx, ess in enumerate(esss)}
    sop_dict = {sop.ID: idx for idx, sop in enumerate(sops)}
    nop_dict = {nop.ID: idx for idx, nop in enumerate(nops)}

    # 2.【聚合所有充电站的数据】

    # 初始化用于聚合所有站数据的列表
    all_present_cars_list = []
    all_boc_initial_list = []
    all_original_envs = []

    # 这是最关键的数据结构：一个将“全局充电桩索引”映射到其“母线ID”的字典
    spot_to_bus_map = {}
    ev_info = {}  #用于存储所有EV的详细信息
    ev_count = 0
    total_spots_count = 0

    # 时间转换参数
    start_hour = gui_params['start_hour']
    end_hour = gui_params['end_hour']
    step_minutes = gui_params['step_minutes']
    steps_per_hour = 60 // step_minutes

    print("正在聚合处理多个充电站的场景数据...")
    for station in stations_list:
        original_env = station.get_scenario_for_baseline()
        all_original_envs.append(original_env)

        # --- a. 重采样 station_present_cars ---
        original_present_cars = original_env.Invalues['present_cars']
        resampled_present_cars_full_day = np.repeat(original_present_cars, steps_per_hour, axis=1)
        start_step = start_hour * steps_per_hour
        end_step = end_hour * steps_per_hour
        station_present_cars = resampled_present_cars_full_day[:, start_step:end_step]
        all_present_cars_list.append(station_present_cars)

        # --- b. 重采样 station_boc_initial ---
        original_boc_initial = original_env.Invalues["BOC"]
        original_time_points = np.arange(original_boc_initial.shape[1])
        new_time_points = np.linspace(start_hour, end_hour, time_steps + 1)
        station_boc_initial = np.zeros((station.num_spots, time_steps + 1))
        for i in range(station.num_spots):
            station_boc_initial[i, :] = np.interp(new_time_points, original_time_points, original_boc_initial[i, :])
        all_boc_initial_list.append(station_boc_initial)

        # --- c. 【新增】处理 ev_info 和 spot_to_bus_map ---
        original_arrival_times = original_env.Invalues['ArrivalT']
        original_departure_times = original_env.Invalues['DepartureT']

        for local_spot_idx in range(station.num_spots):
            global_spot_idx = total_spots_count + local_spot_idx
            spot_to_bus_map[global_spot_idx] = station.bus_id

            # 遍历当前充电桩上的所有充电会话
            for i, (t_arr, t_dep) in enumerate(
                    zip(original_arrival_times[local_spot_idx], original_departure_times[local_spot_idx])):
                # 判断该会话是否在我们的仿真窗口内
                if t_arr < end_hour and t_dep > start_hour:
                    effective_arrival_hour = max(t_arr, start_hour)
                    effective_departure_hour = min(t_dep, end_hour)

                    # 换算为时间步索引
                    new_arrival_step = int(round((effective_arrival_hour - start_hour) * steps_per_hour))
                    new_departure_step = int(round((effective_departure_hour - start_hour) * steps_per_hour))

                    # 确保 arrival < departure
                    if new_arrival_step >= new_departure_step:
                        continue

                    # 创建一个全局唯一的EV ID
                    ev_id = f"EV_ST{station.station_id}_SP{global_spot_idx}_S{i}"

                    # 获取该车到达时的初始电量
                    initial_soc_at_arrival = station_boc_initial[
                        local_spot_idx, new_arrival_step] if new_arrival_step < time_steps + 1 else 0

                    # 存入ev_info字典
                    ev_info[ev_id] = {
                        "spot": global_spot_idx,
                        "arrival": new_arrival_step,
                        "departure": new_departure_step,
                        "initial_boc": initial_soc_at_arrival
                    }
                    ev_count += 1

        total_spots_count += station.num_spots

        # d. 将所有站的数据垂直堆叠成一个大的Numpy数组
    present_cars = np.vstack(all_present_cars_list)
    boc_initial = np.vstack(all_boc_initial_list)
    print(f"数据聚合完成，总充电桩: {total_spots_count}, 总EV会话: {ev_count}")

    # (由于EV物理参数相同，我们只取第一个作为代表)
    ev_capacity = all_original_envs[0].EV_Param['EV_capacity']
    original_env = all_original_envs[0]

    # 从Excel文件加载电价数据（$/puh）
    price = load_electricity_price(gui_params=gui_params)

    # 定义时间步和索引集
    model.T = RangeSet(0, time_steps - 1)  # 时间步 t=0 to t=23
    model.T_plus1 = RangeSet(0, time_steps)  # 用于 SOC 更新，t=0 to t=24
    model.Buses = Set(initialize=[bus.ID for bus in buses])
    model.Lines = Set(initialize=[line.ID for line in lines])
    model.Gens = Set(initialize=[gen.ID for gen in gens])
    model.PVWs = Set(initialize=[pvw.ID for pvw in pvws])
    model.ESSs = Set(initialize=[ess.ID for ess in esss])
    model.SOPs = Set(initialize=[sop.ID for sop in sops])
    model.NOPs = Set(initialize=[nop.ID for nop in nops])
    # 新增一个基于 ev_info 的集合，代表所有独立的充电事件
    model.EVs = Set(initialize=ev_info.keys())
    model.Spots = RangeSet(0, total_spots_count - 1)# 充电桩编号

    # 变量定义
    # # 添加决策变量：从上级电网流入slack bus的功率 (pu)
    # def grid_inflow_p_bounds(model, t):
    #     return (-float('inf'), float('inf'))  # 假设购电量不受上限限制，允许流入流出
    # 将电网交互功率定义为非负实数，强制其只能是购电（>=0），不能售电(<0)
    model.grid_inflow_p = Var(model.T, domain=NonNegativeReals, name="grid_inflow_p")    # 母线电压 (V)
    model.v = Var(model.Buses, model.T,
                  bounds=lambda model, bus_id, t: (grid.Bus(bus_id).MinV, grid.Bus(bus_id).MaxV))

    # 线路有功功率 (P)
    model.P = Var(model.Lines, model.T)

    # 线路无功功率 (Q)
    model.Q = Var(model.Lines, model.T)

    # 发电机有功功率 (pg)
    def pg_bounds(model, gen_id, t):
        gen = next(g for g in gens if g.ID == gen_id)
        pmin = gen.Pmin(t) if callable(gen.Pmin) else gen.Pmin
        pmax = gen.Pmax(t) if callable(gen.Pmax) else gen.Pmax

        # 直接从发电机对象上读取我们设置的真实上限
        final_pmax = min(pmax, gen.RealisticPmax) if pmax is not None else gen.RealisticPmax

        return (pmin, final_pmax)

    model.pg = Var(model.Gens, model.T, bounds=pg_bounds)

    # 发电机无功功率 (qg)
    def qg_bounds(model, gen_id, t):
        gen = next(g for g in gens if g.ID == gen_id)
        qmin = gen.Qmin(t) if callable(gen.Qmin) else (gen.Qmin if gen.Qmin is not None else -float('inf'))
        qmax = gen.Qmax(t) if callable(gen.Qmax) else (gen.Qmax if gen.Qmax is not None else float('inf'))
        return (qmin, qmax)

    model.qg = Var(model.Gens, model.T, bounds=qg_bounds)

    # 充电桩充电/放电功率 (pspot)，正值表示充电，负值表示放电 (V2G)
    def pspot_bounds(model, spot, t):
        if present_cars[spot, t] == 1:
            return (-original_env.EV_Param['discharging_rate'] / (sb_mva * 1000),
                    original_env.EV_Param['charging_rate'] / (sb_mva * 1000))
        return (0, 0)
    # model.pspot = Var(model.Spots, model.T, bounds=pspot_bounds)
    # model.boc_spot = Var(model.Spots, model.T_plus1, bounds=(0, 1))
    def pev_charge_bounds(model, ev_id, t):
        info = ev_info[ev_id]
        # 只有当车辆在场时，才允许有充电功率
        if present_cars[info['spot'], t] == 1:
            # 最大充电功率，转换为pu单位
            max_charge_pu = original_env.EV_Param['charging_rate'] / (sb_mva * 1000)
            return (0, max_charge_pu)
        return (0, 0)  # 不在场时，充电功率为0

    def pev_discharge_bounds(model, ev_id, t):
        info = ev_info[ev_id]
        # 只有当车辆在场时，才允许有放电功率
        if present_cars[info['spot'], t] == 1:
            # 最大放电功率，转换为pu单位
            max_discharge_pu = original_env.EV_Param['discharging_rate'] / (sb_mva * 1000)
            return (0, max_discharge_pu)
        return (0, 0)  # 不在场时，放电功率为0

    model.pev_charge = Var(model.EVs, model.T, bounds=pev_charge_bounds)
    model.pev_discharge = Var(model.EVs, model.T, bounds=pev_discharge_bounds)
    model.ev_charge_or_discharge = Var(model.EVs, model.T, domain=Binary)
    model.pev = Var(model.EVs, model.T, bounds=lambda m, ev_id, t: pspot_bounds(m, ev_info[ev_id]['spot'], t))
    model.boc_ev = Var(model.EVs, model.T_plus1, bounds=(0, 1))

    # 光伏和风电出力 (pvw_p)，考虑最大出力限制
    def pvw_p_bounds(model, pvw_id, t):
        pvw = next(p for p in pvws if p.ID == pvw_id)
        p_max = pvw.P(t * 3600) if callable(pvw.P) else pvw.P
        return (0, p_max)

    model.pvw_p = Var(model.PVWs, model.T, bounds=pvw_p_bounds)

    def pvw_q_bounds(model, pvw_id, t):
        pvw = next(p for p in pvws if p.ID == pvw_id)
        p_max = pvw.P(t * 3600) if callable(pvw.P) else pvw.P
        return (-p_max * np.sqrt(1 - pvw.PF ** 2) / pvw.PF if pvw.PF != 0 else -p_max,
                p_max * np.sqrt(1 - pvw.PF ** 2) / pvw.PF if pvw.PF != 0 else p_max)

    model.pvw_q = Var(model.PVWs, model.T, bounds=pvw_q_bounds)

    # 储能系统充电/放电功率 (ess_p)，正值表示充电，负值表示放电
    def ess_p_bounds(model, ess_id, t):
        ess = next(e for e in esss if e.ID == ess_id)
        # 只根据储能单元自身的物理最大充放电功率来设定边界
        return (-ess.MaxPd, ess.MaxPc)


    model.ess_p = Var(model.ESSs, model.T, bounds=ess_p_bounds)

    def ess_charge_bounds(model, ess_id, t):
        ess = next(e for e in esss if e.ID == ess_id)
        return (0, ess.MaxPc)

    model.ess_charge = Var(model.ESSs, model.T, bounds=ess_charge_bounds)

    def ess_discharge_bounds(model, ess_id, t):
        ess = next(e for e in esss if e.ID == ess_id)
        return (0, ess.MaxPd)

    model.ess_discharge = Var(model.ESSs, model.T, bounds=ess_discharge_bounds)


    def ess_soc_bounds(model, ess_id, t):
        ess = next(e for e in esss if e.ID == ess_id)
        return (0, ess.Cap)

    model.ess_soc = Var(model.ESSs, model.T_plus1, bounds=ess_soc_bounds)

    # SOP 功率流变量
    def sop_p1_bounds(model, sop_id, t):
        sop = next(s for s in sops if s.ID == sop_id)
        return (-sop.PMax, sop.PMax) if sop.active else (0, 0)

    model.sop_p1 = Var(model.SOPs, model.T, bounds=sop_p1_bounds)

    def sop_q1_bounds(model, sop_id, t):
        sop = next(s for s in sops if s.ID == sop_id)
        return (-sop.QMax, sop.QMax) if sop.active else (0, 0)

    model.sop_q1 = Var(model.SOPs, model.T, bounds=sop_q1_bounds)

    def sop_loss_bounds(model, sop_id, t):
        return (0, float('inf')) if next(s for s in sops if s.ID == sop_id).active else (0, 0)

    model.sop_loss = Var(model.SOPs, model.T, bounds=sop_loss_bounds)

    # NOP 状态变量（0 表示开，1 表示闭合）和功率流
    model.nop_status = Var(model.NOPs, model.T, domain=Binary)
    model.nop_p = Var(model.NOPs, model.T)
    model.nop_q = Var(model.NOPs, model.T)

    # 松弛变量，用于潮流平衡
    model.SlackP = Var(model.Buses, model.T)
    model.SlackQ = Var(model.Buses, model.T)

    model.unfull_energy_kwh = Var(ev_info.keys(), domain=NonNegativeReals)

    # 为SOP容量约束添加松弛变量
    model.sop_capacity_slack = Var(model.SOPs, model.T, domain=NonNegativeReals)
    # 为NOP电压约束添加松弛变量
    model.nop_v_slack_pos = Var(model.NOPs, model.T, domain=NonNegativeReals)
    model.nop_v_slack_neg = Var(model.NOPs, model.T, domain=NonNegativeReals)
    return (model, ev_info, price, present_cars, boc_initial, original_env,
            ev_capacity, buses, lines, gens, pvws, esss, sops, nops,
            bus_dict, line_dict, gen_dict, pvw_dict, ess_dict, sop_dict,
            nop_dict, grid, spot_to_bus_map)

def add_constraints(model, ev_info, present_cars, boc_initial, original_env, ev_capacity, buses, lines, gens, pvws,
                    esss, sops, nops, bus_dict, line_dict, gen_dict, pvw_dict, ess_dict, sop_dict, nop_dict, grid,
                    time_steps, spot_to_bus_map, gui_params): # <--- 在末尾添加 gui_params
    """
    (多充电站版)
    添加 Pyomo 模型的约束条件，特别是重构了潮流平衡以支持多个充电站。
    """
    sb_mva = grid.SB
    #  pev功率分离约束
    def pev_split_rule(model, ev_id, t):
        info = ev_info[ev_id]
        if info['arrival'] <= t < info['departure']:
            return model.pev[ev_id, t] == model.pev_charge[ev_id, t] - model.pev_discharge[ev_id, t]
        else:
            # 如果车辆不在场，其个体功率为0
            model.pev[ev_id, t].fix(0)
            return Constraint.Skip

    model.pev_split_constr = Constraint(model.EVs, model.T, rule=pev_split_rule)

    # BOC更新约束 (基于EV)
    def boc_ev_update_rule(model, ev_id, t):
        """
        【v3.0 最终修复版】使用正确的物理公式更新EV BOC，包含时间步长。
        """
        info = ev_info[ev_id]
        step_duration_h = gui_params['step_minutes'] / 60.0  # 获取时间步长（小时）

        # 该规则只在车辆的停留时间内有效
        if info['arrival'] <= t < info['departure']:
            # 计算充、放电导致的SOC变化量，注意要乘以时间
            charge_change = (model.pev_charge[ev_id, t] * sb_mva * 1000 * original_env.EV_Param[
                'charging_effic'] * step_duration_h) / ev_capacity
            discharge_change = (model.pev_discharge[ev_id, t] * sb_mva * 1000 * step_duration_h) / (
                        ev_capacity * original_env.EV_Param['discharging_effic'])

            # 判断是否为车辆到达的第一个时间步
            if t == info['arrival']:
                # 到达时刻，BOC从初始值开始更新
                return model.boc_ev[ev_id, t + 1] == info['initial_boc'] + charge_change - discharge_change
            else:
                # 非到达时刻，BOC从上一时刻继承并更新
                return model.boc_ev[ev_id, t + 1] == model.boc_ev[ev_id, t] + charge_change - discharge_change

        # 如果车辆在当前时间步刚好离开，我们定义其当前BOC等于上一步的最终值，以便读取
        # 注意: Pyomo的约束是针对 t in model.T (0 to 22/23), 而boc_ev有 T+1
        if t > 0 and t == info['departure']:
            return model.boc_ev[ev_id, t] == model.boc_ev[ev_id, t]

        return Constraint.Skip

    model.boc_ev_update_constr = Constraint(model.EVs, model.T, rule=boc_ev_update_rule)

    # 约束1：只有当开关 ev_charge_or_discharge 为 1 时，充电功率 pev_charge才能大于0
    def ev_charge_limit_rule(model, ev_id, t):
        max_charge_pu = original_env.EV_Param['charging_rate'] / (sb_mva * 1000)
        return model.pev_charge[ev_id, t] <= model.ev_charge_or_discharge[ev_id, t] * max_charge_pu

    model.ev_charge_limit_constr = Constraint(model.EVs, model.T, rule=ev_charge_limit_rule)

    # 约束2：只有当开关 ev_charge_or_discharge 为 0 时，放电功率 pev_discharge才能大于0
    def ev_discharge_limit_rule(model, ev_id, t):
        max_discharge_pu = original_env.EV_Param['discharging_rate'] / (sb_mva * 1000)
        return model.pev_discharge[ev_id, t] <= (1 - model.ev_charge_or_discharge[ev_id, t]) * max_discharge_pu

    model.ev_discharge_limit_constr = Constraint(model.EVs, model.T, rule=ev_discharge_limit_rule)

    # 3. 储能系统电量更新约束
    def initial_ess_soc_rule(model, ess_id):
        ess = next(e for e in esss if e.ID == ess_id)
        return model.ess_soc[ess_id, 0] == ess.SOC * ess.Cap

    model.initial_ess_soc_constr = Constraint(model.ESSs, rule=initial_ess_soc_rule)

    def ess_soc_update_rule(model, ess_id, t):
        """
        【v2.0 修复版】使用正确的物理公式更新ESS SOC。
        """
        ess = next(e for e in esss if e.ID == ess_id)
        step_duration_h = gui_params['step_minutes'] / 60.0

        # 【修复】SOC的变化量 = (功率 * 时间) / 容量
        charge_change = (model.ess_charge[ess_id, t] * ess.EC * step_duration_h) / ess.Cap
        discharge_change = (model.ess_discharge[ess_id, t] / ess.ED * step_duration_h) / ess.Cap

        return model.ess_soc[ess_id, t + 1] == model.ess_soc[ess_id, t] + charge_change - discharge_change

    model.ess_soc_update_constr = Constraint(model.ESSs, model.T, rule=ess_soc_update_rule)

    def ess_p_split_rule(model, ess_id, t):
        return model.ess_p[ess_id, t] == model.ess_charge[ess_id, t] - model.ess_discharge[ess_id, t]

    model.ess_p_split_constr = Constraint(model.ESSs, model.T, rule=ess_p_split_rule)


    # 新增约束：防止 ESS 同时充电和放电
    model.ess_charge_or_discharge = Var(model.ESSs, model.T, domain=Binary)

    def ess_charge_limit_rule(model, ess_id, t):
        ess = next(e for e in esss if e.ID == ess_id)
        return model.ess_charge[ess_id, t] <= model.ess_charge_or_discharge[ess_id, t] * ess.MaxPc

    model.ess_charge_limit_constr = Constraint(model.ESSs, model.T, rule=ess_charge_limit_rule)

    def ess_discharge_limit_rule(model, ess_id, t):
        ess = next(e for e in esss if e.ID == ess_id)
        return model.ess_discharge[ess_id, t] <= (1 - model.ess_charge_or_discharge[ess_id, t]) * ess.MaxPd

    model.ess_discharge_limit_constr = Constraint(model.ESSs, model.T, rule=ess_discharge_limit_rule)

    # 4. 光伏和风电无功出力约束
    def pvw_q_rel_rule(model, pvw_id, t):
        pvw = next(p for p in pvws if p.ID == pvw_id)
        tan_phi = np.sqrt(1 - pvw.PF ** 2) / pvw.PF if pvw.PF != 0 else 0
        return model.pvw_q[pvw_id, t] == model.pvw_p[pvw_id, t] * tan_phi

    model.pvw_q_rel_constr = Constraint(model.PVWs, model.T, rule=pvw_q_rel_rule)

    # 5. 潮流平衡约束（有功和无功）
    @model.Constraint(model.Buses, model.T, doc="有功功率平衡约束")
    def p_balance_rule(model, bus_id, t):
        # 【修复】根据时间步索引t，计算正确的秒数
        time_in_seconds = t * gui_params['step_minutes'] * 60

        # --- 注入项 (Sources) ---
        power_injections = (
                sum(model.P[l.ID, t] for l in grid.LinesOfTBus(bus_id, only_active=True)) +
                sum(model.pg[g.ID, t] for g in grid.GensAtBus(bus_id) if g.ID in model.Gens) +
                sum(model.pvw_p[pvw.ID, t] for pvw in pvws if pvw.BusID == bus_id) +
                sum(model.nop_p[nop.ID, t] for nop in nops if nop.Bus2 == bus_id)
        )
        power_injections += sum(model.sop_p1[sop.ID, t] - model.sop_loss[sop.ID, t]
                                for sop in sops if sop.Bus2 == bus_id)

        # --- 流出项 (Ejections) ---
        power_ejections = (
                sum(model.P[l.ID, t] for l in grid.LinesOfFBus(bus_id, only_active=True)) +
                grid.Bus(bus_id).Pd(time_in_seconds) +  # 【修复】使用正确的秒数来获取负荷
                sum(model.nop_p[nop.ID, t] for nop in nops if nop.Bus1 == bus_id) +
                sum(model.pev[ev_id, t] for ev_id, info in ev_info.items() if
                    info['spot'] in spot_to_bus_map and spot_to_bus_map[info['spot']] == bus_id) +
                sum(model.ess_p[ess.ID, t] for ess in esss if ess.BusID == bus_id)
        )
        power_ejections += sum(model.sop_p1[sop.ID, t] for sop in sops if sop.Bus1 == bus_id)

        # --- 平衡方程 ---
        if bus_id == CORE_PARAMS["slack_bus"]:
            return power_injections + model.grid_inflow_p[t] == power_ejections
        else:
            return power_injections + model.SlackP[bus_id, t] == power_ejections

    @model.Constraint(model.Buses, model.T, doc="无功功率平衡约束")
    def q_balance_rule(model, bus_id, t):
        # 【修复】根据时间步索引t，计算正确的秒数
        time_in_seconds = t * gui_params['step_minutes'] * 60

        q_injections = (
                sum(model.Q[l.ID, t] for l in grid.LinesOfTBus(bus_id, only_active=True)) +
                sum(model.qg[g.ID, t] for g in grid.GensAtBus(bus_id) if g.ID in model.Gens) +
                sum(model.pvw_q[pvw.ID, t] for pvw in pvws if pvw.BusID == bus_id) +
                sum(model.sop_q1[sop.ID, t] for sop in sops if sop.Bus2 == bus_id) +
                sum(model.nop_q[nop.ID, t] for nop in nops if nop.Bus2 == bus_id)
        )
        q_ejections = (
                sum(model.Q[l.ID, t] for l in grid.LinesOfFBus(bus_id, only_active=True)) +
                grid.Bus(bus_id).Qd(time_in_seconds) +  # 【修复】使用正确的秒数来获取负荷
                sum(model.sop_q1[sop.ID, t] for sop in sops if sop.Bus1 == bus_id) +
                sum(model.nop_q[nop.ID, t] for nop in nops if nop.Bus1 == bus_id)
        )

        if bus_id == CORE_PARAMS["slack_bus"]:
            return q_injections == q_ejections
        else:
            return q_injections + model.SlackQ[bus_id, t] == q_ejections
    # 6. 电压更新约束（线性化 DistFlow 模型）
    def v_update_rule(model, line_id, t):
        line = next(l for l in lines if l.ID == line_id)
        f_bus = line.fBus
        t_bus = line.tBus
        r_pu = line.R
        x_pu = line.X
        return model.v[t_bus, t] == model.v[f_bus, t] - (r_pu * model.P[line_id, t] + x_pu * model.Q[line_id, t])

    model.v_update_constr = Constraint(model.Lines, model.T, rule=v_update_rule)

    # 7. SOP 容量和损耗约束
    def sop_capacity_rule(model, sop_id, t):
        sop = next(s for s in sops if s.ID == sop_id)
        if sop.active:
            # 允许在支付代价的情况下，用松弛变量来超出限制
            return model.sop_p1[sop_id, t] ** 2 + model.sop_q1[sop_id, t] ** 2 <= sop.PMax ** 2 + \
                model.sop_capacity_slack[sop_id, t]
        return Constraint.Skip

    model.sop_capacity_constr = Constraint(model.SOPs, model.T, rule=sop_capacity_rule)

    def sop_loss_rule(model, sop_id, t):
        sop = next(s for s in sops if s.ID == sop_id)
        if sop.active and sop.PMax > 0:
            # 修正：让损耗只与有功功率P相关，移除Q的影响
            return model.sop_loss[sop_id, t] >= sop.LossCoeff * (
                        model.sop_p1[sop_id, t] ** 2) / sop.PMax ** 2
        return Constraint.Skip

    model.sop_loss_constr = Constraint(model.SOPs, model.T, rule=sop_loss_rule)

    # 8. NOP 功率流和电压更新约束
    M = 1000  # 大 M 值，用于逻辑约束

    def nop_p_limit1_rule(model, nop_id, t):
        return model.nop_p[nop_id, t] <= M * model.nop_status[nop_id, t]

    model.nop_p_limit1_constr = Constraint(model.NOPs, model.T, rule=nop_p_limit1_rule)

    def nop_p_limit2_rule(model, nop_id, t):
        return model.nop_p[nop_id, t] >= -M * model.nop_status[nop_id, t]

    model.nop_p_limit2_constr = Constraint(model.NOPs, model.T, rule=nop_p_limit2_rule)

    def nop_q_limit1_rule(model, nop_id, t):
        return model.nop_q[nop_id, t] <= M * model.nop_status[nop_id, t]

    model.nop_q_limit1_constr = Constraint(model.NOPs, model.T, rule=nop_q_limit1_rule)

    def nop_q_limit2_rule(model, nop_id, t):
        return model.nop_q[nop_id, t] >= -M * model.nop_status[nop_id, t]

    model.nop_q_limit2_constr = Constraint(model.NOPs, model.T, rule=nop_q_limit2_rule)

    def nop_v_unified_rule(model, nop_id, t):
        nop = next(n for n in nops if n.ID == nop_id)
        # 如果NOP关闭 (status=1)，则电压关系必须满足(或支付松弛代价)
        # 如果NOP打开 (status=0)，则此约束不生效 (通过大M法实现)
        v_diff = model.v[nop.Bus2, t] - model.v[nop.Bus1, t]
        power_term = (nop.R * model.nop_p[nop_id, t] + nop.X * model.nop_q[nop_id, t])

        expression = v_diff - power_term - (model.nop_v_slack_pos[nop_id, t] - model.nop_v_slack_neg[nop_id, t])

        # 使用两个不等式配合大M法来模拟 if-then
        model.nop_v_upper_constr[nop_id, t] = expression <= M * (1 - model.nop_status[nop_id, t])
        model.nop_v_lower_constr[nop_id, t] = expression >= -M * (1 - model.nop_status[nop_id, t])
        return Constraint.Skip  # 主规则返回Skip，因为我们动态添加了具名约束

    def ev_unfull_energy_rule(model, ev_id):
        info = ev_info[ev_id]
        departure = info["departure"]
        final_soc = model.boc_ev[ev_id, departure]
        return model.unfull_energy_kwh[ev_id] >= (1.00 - final_soc) * ev_capacity

    model.ev_unfull_energy_constr = Constraint(model.EVs, rule=ev_unfull_energy_rule)

    # 10. 线路有功功率上限约束（简化容量限制）
    def line_p_limit_rule(model, line_id, t):
        max_p = 5.0  # 假设每条线路最大有功功率为 3.0 pu，可根据实际情况调整
        return model.P[line_id, t] <= max_p

    model.line_p_limit_constr = Constraint(model.Lines, model.T, rule=line_p_limit_rule)

    def line_p_lower_limit_rule(model, line_id, t):
        min_p = -5.0  # 假设每条线路最小有功功率为 -3.0 pu，可根据实际情况调整
        return model.P[line_id, t] >= min_p

    model.line_p_lower_limit_constr = Constraint(model.Lines, model.T, rule=line_p_lower_limit_rule)

    def fix_slack_bus_voltage_rule(model, t):
        """此规则将松弛母线的电压在所有时间步固定为1.0 pu。"""
        return model.v[CORE_PARAMS["slack_bus"], t] == 1.0

    model.slack_bus_voltage_constraint = Constraint(
        model.T,
        rule=fix_slack_bus_voltage_rule,
        doc="Fix slack bus voltage to 1.0 pu for all timesteps"
    )
    return model

def define_objective_and_solve(model, ev_info, price, buses, gens, pvws, esss, sops, nops, grid,
                               time_steps, solver_name, gui_params):
    """
    定义目标函数，并求解模型
    """

    # 最终修正版：统一了发电机成本模型，并修正了所有成本项的单位。
    def objective_rule(model):
        step_duration_h = gui_params['step_minutes'] / 60.0
        sb_mva = grid.SB
        # 从config中读取参数
        penalties = BASELINE_PARAMS["penalty_factors"]
        ess_cost_factor = BASELINE_PARAMS["ess_degradation_cost"]
        ev_penalty_factor = penalties["ev_not_full_penalty"]
        # 购电成本 (单位: 元)
        grid_purchase_cost = sum(price[t] * model.grid_inflow_p[t] * sb_mva * 1000 * step_duration_h for t in model.T)

        # 发电成本 (单位: 元) - 使用与RL环境一致的二次成本模型
        generation_cost = sum(
            (grid.Gen(g_id).CostA(t) * (model.pg[g_id, t] * sb_mva) ** 2 +
             grid.Gen(g_id).CostB(t) * (model.pg[g_id, t] * sb_mva) +
             grid.Gen(g_id).CostC(t)) * step_duration_h
            for g_id in model.Gens for t in model.T
        )

        # SOP损耗成本 (单位: 元)
        sop_loss_cost = sum(price[t] * model.sop_loss[sop_id, t] * sb_mva * 1000 * step_duration_h
                            for sop_id in model.SOPs for t in model.T)

        # ESS放电成本 (单位: 元)
        ess_discharge_cost = sum(ess_cost_factor * model.ess_discharge[ess_id, t] * sb_mva * 1000 * step_duration_h
                                 for ess_id in model.ESSs for t in model.T)

        # 各类惩罚项
        slack_penalty = penalties["slack_power_penalty"] * sum(model.SlackP[b, t] ** 2 + model.SlackQ[b, t] ** 2 for b in model.Buses for t in model.T)
        ev_not_full_penalty = ev_penalty_factor * sum(model.unfull_energy_kwh.values())
        sop_slack_penalty = penalties["sop_capacity_penalty"] * sum(model.sop_capacity_slack.values())
        nop_slack_penalty = penalties["nop_voltage_penalty"] * sum(model.nop_v_slack_pos.values()) + \
                            penalties["nop_voltage_penalty"] * sum(model.nop_v_slack_neg.values())

        # 最终目标: 最小化所有成本与惩罚之和
        obj = (grid_purchase_cost + generation_cost + sop_loss_cost + ess_discharge_cost +
               slack_penalty + ev_not_full_penalty + sop_slack_penalty + nop_slack_penalty)
        return obj

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # 求解模型 (保持不变)
    solver = SolverFactory(solver_name)
    solver.options['TimeLimit'] = 300
    solver.options['OutputFlag'] = 1
    results = solver.solve(model, tee=True)

    # 检查求解结果
    if (results.solver.status in [SolverStatus.ok, SolverStatus.warning] and
            results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.globallyOptimal,
                                                     TerminationCondition.feasible]):
        model.solutions.load_from(results)
        solve_result = GridSolveResult.OK
        print("求解成功！正在提取结果...")

        # 直接从模型中重新计算，确保与优化目标完全一致
        step_duration_h = gui_params['step_minutes'] / 60.0
        sb_mva = grid.SB
        ess_cost_factor = 0.0

        grid_purchase_cost_val = sum(
            price[t] * value(model.grid_inflow_p[t]) * sb_mva * 1000 * step_duration_h for t in model.T)

        generation_cost_val = sum(
            (value(grid.Gen(g_id).CostA(t)) * (value(model.pg[g_id, t]) * sb_mva) ** 2 +
             value(grid.Gen(g_id).CostB(t)) * (value(model.pg[g_id, t]) * sb_mva) +
             value(grid.Gen(g_id).CostC(t))) * step_duration_h
            for g_id in model.Gens for t in model.T
        )

        sop_loss_cost_val = sum(price[t] * value(model.sop_loss[sop_id, t]) * sb_mva * 1000 * step_duration_h
                                for sop_id in model.SOPs for t in model.T)

        ess_discharge_cost_val = sum(
            ess_cost_factor * value(model.ess_discharge[ess_id, t]) * sb_mva * 1000 * step_duration_h
            for ess_id in model.ESSs for t in model.T)

        slack_penalty_val = 1e7 * sum(value(model.SlackP[b, t]) ** 2 + value(model.SlackQ[b, t]) ** 2
                                      for b in model.Buses for t in model.T)

        # 统一使用配置文件中的惩罚因子进行结果验证
        penalties = BASELINE_PARAMS["penalty_factors"]
        ev_not_full_penalty_val = penalties["ev_not_full_penalty"] * sum(
            value(v) for v in model.unfull_energy_kwh.values())
        sop_slack_penalty_val = penalties["sop_capacity_penalty"] * sum(
            value(v) for v in model.sop_capacity_slack.values())
        nop_slack_penalty_val = penalties["nop_voltage_penalty"] * sum(
            value(v) for v in model.nop_v_slack_pos.values()) + \
                                penalties["nop_voltage_penalty"] * sum(value(v) for v in model.nop_v_slack_neg.values())

        baseline_data = {
            "objective_value": value(model.objective),
            "grid_purchase_cost": grid_purchase_cost_val,
            "generation_cost": generation_cost_val,
            "sop_loss_cost": sop_loss_cost_val,
            "ess_discharge_cost": ess_discharge_cost_val,
            "slack_penalty": slack_penalty_val,
            "ev_not_full_penalty": ev_not_full_penalty_val,
            "sop_slack_penalty": sop_slack_penalty_val,
            "nop_slack_penalty": nop_slack_penalty_val,
            "sop_slacks": {sop.ID: [value(model.sop_capacity_slack[sop.ID, t]) for t in model.T] for sop in sops},
            "grid_inflow_p": [value(model.grid_inflow_p[t]) for t in model.T],
            "gen_powers": {gen_id: [value(model.pg[gen_id, t]) for t in model.T] for gen_id in model.Gens},
            "slack_powers": {b: [value(model.SlackP[b, t]) for t in model.T] for b in model.Buses},
            "bus_voltages": {},
            "ev_info": {},
            "total_ev_count": len(ev_info),
            "charged_ev_count": 0,
            "pvw_powers": {},
            "ess_powers": {},
            "ess_soc": {},
            "sop_flows": {},
            "nop_status": {},
            "nop_flows": {},
            # 【新增】初始化spot_powers和ev_powers
            "spot_powers": {},
            "ev_powers": {}
        }

        # 提取物理量
        time_steps = len(list(model.T))

        for bus_id in model.Buses:
            baseline_data["bus_voltages"][bus_id] = [value(model.v[bus_id, t]) for t in model.T]

        baseline_data["line_powers"] = {}
        for line_id in model.Lines:
            baseline_data["line_powers"][line_id] = [value(model.P[line_id, t]) for t in model.T]

        # --- 【核心修复区域】 ---
        # 1. 【修改】不再直接读取 pspot，而是先提取新的 pev (分车辆功率)
        for ev_id in model.EVs:
            baseline_data["ev_powers"][ev_id] = [value(model.pev[ev_id, t]) for t in model.T]

        # 2. 【新增】将分车辆功率（pev）聚合为分充电桩功率（spot_powers），以兼容下游分析
        num_spots = len(list(model.Spots))
        baseline_data["spot_powers"] = {spot_idx: [0.0] * time_steps for spot_idx in range(num_spots)}
        for ev_id, info in ev_info.items():
            spot_id = info['spot']
            ev_powers_over_time = baseline_data["ev_powers"][ev_id]
            for t in range(time_steps):
                baseline_data["spot_powers"][spot_id][t] += ev_powers_over_time[t]

        # 3. 【修改】从新的 boc_ev 变量中提取BOC，而不是旧的 boc_spot
        charged_count = 0
        for ev_id, info in ev_info.items():
            departure = info["departure"]
            # 确保离开时间在索引范围内
            final_soc = value(model.boc_ev[ev_id, departure]) if departure <= time_steps else 0
            standards = EVALUATION_CONFIG["standards"]  # 确保在作用域内可访问
            is_charged = final_soc >= standards["ev_charged_soc_threshold"]
            if is_charged:
                charged_count += 1
            baseline_data["ev_info"][ev_id] = {"initial_boc": info["initial_boc"], "final_boc": final_soc,
                                               "charged": is_charged}
        baseline_data["charged_ev_count"] = charged_count
        # --- 【核心修复结束】 ---

        for pvw_id in model.PVWs:
            baseline_data["pvw_powers"][pvw_id] = [value(model.pvw_p[pvw_id, t]) for t in model.T]

        for ess_id in model.ESSs:
            baseline_data["ess_powers"][ess_id] = [value(model.ess_p[ess_id, t]) for t in model.T]
            baseline_data["ess_soc"][ess_id] = [value(model.ess_soc[ess_id, t]) for t in model.T_plus1]

        for sop_id in model.SOPs:
            if next(s for s in sops if s.ID == sop_id).active:
                baseline_data["sop_flows"][sop_id] = {"P1": [value(model.sop_p1[sop_id, t]) for t in model.T],
                                                      "Q1": [value(model.sop_q1[sop_id, t]) for t in model.T],
                                                      "Loss": [value(model.sop_loss[sop_id, t]) for t in model.T]}

        for nop_id in model.NOPs:
            baseline_data["nop_status"][nop_id] = [value(model.nop_status[nop_id, t]) for t in model.T]
            baseline_data["nop_flows"][nop_id] = {"P": [value(model.nop_p[nop_id, t]) for t in model.T],
                                                  "Q": [value(model.nop_q[nop_id, t]) for t in model.T]}

        print("结果提取完毕。")
        return solve_result, baseline_data, model

    else:
        solve_result = GridSolveResult.Failed
        print("!!!!!!!! 求解失败 !!!!!!!!")
        print(f"求解器状态: {results.solver.status}")
        print(f"终止条件: {results.solver.termination_condition}")
        return solve_result, {}, None # 失败时返回None


# file: baseline.py

def solve_baseline(grid: Grid, stations_list, gui_params: dict):
    """
    (多充电站版)
    整合模型创建、约束添加、目标定义和求解过程。
    """
    # 从参数字典中提取时间配置
    time_steps = int((gui_params['end_hour'] - gui_params['start_hour']) * (60 // gui_params['step_minutes']))


    # 创建模型和变量，现在会解包出23个值
    model_data = create_baseline_model(grid, stations_list, time_steps, gui_params)
    (model, ev_info, price, present_cars, boc_initial, original_env,
     ev_capacity, buses, lines, gens, pvws, esss, sops, nops,
     bus_dict, line_dict, gen_dict, pvw_dict, ess_dict, sop_dict,
     nop_dict, grid, spot_to_bus_map) = model_data

    # 添加约束
    model = add_constraints(model, ev_info, present_cars, boc_initial, original_env, ev_capacity, buses, lines, gens,
                            pvws, esss, sops, nops, bus_dict, line_dict, gen_dict, pvw_dict, ess_dict, sop_dict,
                            nop_dict, grid, time_steps, spot_to_bus_map, gui_params)


    # 定义目标函数并求解，接收所有返回值
    result, baseline_data, solved_model = define_objective_and_solve(model, ev_info, price, buses, gens, pvws, esss,
                                                                     sops, nops, grid,
                                                                     time_steps, gui_params['solver'], gui_params)

    # 将所有结果（包括模型对象）返回给调用者
    return result, baseline_data, solved_model

