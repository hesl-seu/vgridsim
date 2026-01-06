import os
import sys
import gymnasium as gym
import pandas as pd
from grid_model import create_grid
from baseline import solve_baseline, create_baseline_model, add_constraints, define_objective_and_solve
from config import CORE_PARAMS, TIMESTEPS_PER_EPISODE
from fpowerkit.solbase import GridSolveResult
from fpowerkit.soldss import OpenDSSSolver
from pyomo.environ import value
import pickle

# 确保输出目录存在，设置为项目根目录下的 results/distflow 和 results/opendss
root_dir = os.getcwd()  # 获取当前工作目录（根目录）
distflow_dir = os.path.join(root_dir, "results", "distflow")
opendss_dir = os.path.join(root_dir, "results", "opendss")
if not os.path.exists(distflow_dir):
    os.makedirs(distflow_dir, exist_ok=True)
if not os.path.exists(opendss_dir):
    os.makedirs(opendss_dir, exist_ok=True)


def save_distflow_results(baseline_data, grid, timestep, result, objective_value, model=None):
    """
    保存 Linear DistFlow 的结果，使用 baseline_data 或 model 中的数据。
    :param baseline_data: 从 solve_baseline 返回的结果字典
    :param grid: 配电网对象
    :param timestep: 当前时间步
    :param result: 求解结果状态 (GridSolveResult)
    :param objective_value: 目标值
    :param model: Pyomo 模型对象，用于直接提取数据
    :return: 是否成功保存
    """
    try:
        directory = distflow_dir

        # 保存母线电压结果
        bus_data = []
        if model and hasattr(model, 'Buses') and hasattr(model, 'v'):
            for bus_id in model.Buses:
                bus = grid.Bus(bus_id) if hasattr(grid, 'Bus') and callable(getattr(grid, 'Bus')) else None
                voltage = value(model.v[bus_id, timestep]) if timestep in model.T else 'N/A'
                pd_val = bus.Pd(timestep) if bus and hasattr(bus, 'Pd') and callable(getattr(bus, 'Pd', None)) else (getattr(bus, 'Pd', 'N/A') if bus and hasattr(bus, 'Pd') else 'N/A')
                qd_val = bus.Qd(timestep) if bus and hasattr(bus, 'Qd') and callable(getattr(bus, 'Qd', None)) else (getattr(bus, 'Qd', 'N/A') if bus and hasattr(bus, 'Qd') else 'N/A')
                bus_data.append({
                    'BusID': bus_id,
                    'Voltage_pu': voltage,
                    'Pd_pu': pd_val,
                    'Qd_pu': qd_val
                })
        else:
            for bus_id, voltages in baseline_data["bus_voltages"].items():
                bus = grid.Bus(bus_id) if hasattr(grid, 'Bus') and callable(getattr(grid, 'Bus')) else None
                voltage = voltages[timestep] if timestep < len(voltages) else 'N/A'
                pd_val = bus.Pd(timestep) if bus and hasattr(bus, 'Pd') and callable(getattr(bus, 'Pd', None)) else (getattr(bus, 'Pd', 'N/A') if bus and hasattr(bus, 'Pd') else 'N/A')
                qd_val = bus.Qd(timestep) if bus and hasattr(bus, 'Qd') and callable(getattr(bus, 'Qd', None)) else (getattr(bus, 'Qd', 'N/A') if bus and hasattr(bus, 'Qd') else 'N/A')
                bus_data.append({
                    'BusID': bus_id,
                    'Voltage_pu': voltage,
                    'Pd_pu': pd_val,
                    'Qd_pu': qd_val
                })
        bus_df = pd.DataFrame(bus_data)
        bus_file = os.path.join(directory, f"bus_results_t{timestep}.csv")
        bus_df.to_csv(bus_file, index=False)
        print(f"保存 {bus_file}，数据行数：{len(bus_data)}，绝对路径：{os.path.abspath(bus_file)}")

        # 保存充电桩功率分配结果
        spot_data = []
        for spot, powers in baseline_data["spot_powers"].items():
            power = powers[timestep] if timestep < len(powers) else 0.0
            spot_data.append({
                'SpotID': spot,
                'Power_pu': power
            })
        spot_df = pd.DataFrame(spot_data)
        spot_file = os.path.join(directory, f"spot_results_t{timestep}.csv")
        spot_df.to_csv(spot_file, index=False)
        print(f"保存 {spot_file}，数据行数：{len(spot_data)}，绝对路径：{os.path.abspath(spot_file)}")

        # 保存光伏/风电出力结果
        pvw_data = []
        for pvw_id, powers in baseline_data["pvw_powers"].items():
            power = powers[timestep] if timestep < len(powers) else 0.0
            pvw_data.append({
                'PVWID': pvw_id,
                'Power_pu': power
            })
        pvw_df = pd.DataFrame(pvw_data)
        pvw_file = os.path.join(directory, f"pvw_results_t{timestep}.csv")
        pvw_df.to_csv(pvw_file, index=False)
        print(f"保存 {pvw_file}，数据行数：{len(pvw_data)}，绝对路径：{os.path.abspath(pvw_file)}")

        # 保存储能系统功率和电量结果
        ess_data = []
        for ess_id, powers in baseline_data["ess_powers"].items():
            power = powers[timestep] if timestep < len(powers) else 0.0
            socs = baseline_data["ess_soc"].get(ess_id, [])
            soc = socs[timestep] if timestep < len(socs) else 0.0
            ess_data.append({
                'ESSID': ess_id,
                'Power_pu': power,
                'SOC_puh': soc
            })
        ess_df = pd.DataFrame(ess_data)
        ess_file = os.path.join(directory, f"ess_results_t{timestep}.csv")
        ess_df.to_csv(ess_file, index=False)
        print(f"保存 {ess_file}，数据行数：{len(ess_data)}，绝对路径：{os.path.abspath(ess_file)}")

        # 保存发电机功率结果（如果有 model 对象）
        gen_data = []
        if model and hasattr(model, 'Gens') and hasattr(model, 'pg') and hasattr(model, 'qg'):
            for gen_id in model.Gens:
                gen = next((g for g in grid.Gens if g.ID == gen_id), None) if hasattr(grid, 'Gens') else None
                p_val = value(model.pg[gen_id, timestep]) if timestep in model.T else 0.0
                q_val = value(model.qg[gen_id, timestep]) if timestep in model.T else 0.0
                gen_data.append({
                    'GenID': gen_id,
                    'BusID': gen.BusID if gen else 'N/A',
                    'P_pu': p_val,
                    'Q_pu': q_val
                })
        gen_df = pd.DataFrame(gen_data)
        gen_file = os.path.join(directory, f"gen_results_t{timestep}.csv")
        gen_df.to_csv(gen_file, index=False)
        print(f"保存 {gen_file}，数据行数：{len(gen_data)}，绝对路径：{os.path.abspath(gen_file)}")

        # 保存线路功率结果（如果有 model 对象）
        line_data = []
        if model and hasattr(model, 'Lines') and hasattr(model, 'P') and hasattr(model, 'Q'):
            for line_id in model.Lines:
                line = next((l for l in grid.Lines if l.ID == line_id), None) if hasattr(grid, 'Lines') else None
                p_val = value(model.P[line_id, timestep]) if timestep in model.T else 0.0
                q_val = value(model.Q[line_id, timestep]) if timestep in model.T else 0.0
                line_data.append({
                    'LineID': line_id,
                    'FromBus': line.fBus if line else 'N/A',
                    'ToBus': line.tBus if line else 'N/A',
                    'P_pu': p_val,
                    'Q_pu': q_val,
                    'I_pu': 'N/A'
                })
        line_df = pd.DataFrame(line_data)
        line_file = os.path.join(directory, f"line_results_t{timestep}.csv")
        line_df.to_csv(line_file, index=False)
        print(f"保存 {line_file}，数据行数：{len(line_data)}，绝对路径：{os.path.abspath(line_file)}")

        # 保存求解状态和目标值
        summary_file = os.path.join(directory, f"summary_t{timestep}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Solver: Linear DistFlow\n")
            f.write(f"Result: {result}\n")
            f.write(f"Objective Value: {objective_value:.4f}\n")
        print(f"保存 {summary_file}，绝对路径：{os.path.abspath(summary_file)}")
        return True  # 表示保存成功
    except Exception as e:
        print(f"保存 distflow 结果时发生错误 (时间步 {timestep}): {e}")
        return False  # 表示保存失败


def save_opendss_results(grid, timestep, result, value):
    """
    保存 OpenDSSSolver 的结果。
    :param grid: 配电网对象
    :param timestep: 当前时间步
    :param result: 求解结果状态 (GridSolveResult)
    :param value: 目标值
    :return: 是否成功保存
    """
    try:
        directory = opendss_dir
        # 保存 OpenDSSSolver 的结果
        bus_data = []
        bus_count_updated = 0
        for bus in grid.Buses if hasattr(grid, 'Buses') else []:
            voltage = bus.V if bus.V is not None else 'N/A'
            if bus.V is not None:
                bus_count_updated += 1
            pd_val = bus.Pd(timestep) if hasattr(bus, 'Pd') and callable(getattr(bus, 'Pd', None)) else (getattr(bus, 'Pd', 'N/A') if hasattr(bus, 'Pd') else 'N/A')
            qd_val = bus.Qd(timestep) if hasattr(bus, 'Qd') and callable(getattr(bus, 'Qd', None)) else (getattr(bus, 'Qd', 'N/A') if hasattr(bus, 'Qd') else 'N/A')
            bus_data.append({
                'BusID': bus.ID,
                'Voltage_pu': voltage,
                'Pd_pu': pd_val,
                'Qd_pu': qd_val
            })
        bus_df = pd.DataFrame(bus_data)
        bus_file = os.path.join(directory, f"bus_results_t{timestep}.csv")
        bus_df.to_csv(bus_file, index=False)
        print(f"保存 {bus_file}，数据行数：{len(bus_data)}，电压更新数量：{bus_count_updated}，绝对路径：{os.path.abspath(bus_file)}")

        line_data = []
        line_count_updated = 0
        for line in grid.Lines if hasattr(grid, 'Lines') else []:
            p_val = line.P if line.P is not None else 'N/A'
            q_val = line.Q if line.Q is not None else 'N/A'
            i_val = line.I if line.I is not None else 'N/A'
            if line.P is not None:
                line_count_updated += 1
            line_data.append({
                'LineID': line.ID,
                'FromBus': line.fBus,
                'ToBus': line.tBus,
                'P_pu': p_val,
                'Q_pu': q_val,
                'I_pu': i_val
            })
        line_df = pd.DataFrame(line_data)
        line_file = os.path.join(directory, f"line_results_t{timestep}.csv")
        line_df.to_csv(line_file, index=False)
        print(f"保存 {line_file}，数据行数：{len(line_data)}，功率更新数量：{line_count_updated}，绝对路径：{os.path.abspath(line_file)}")

        gen_data = []
        gen_count_updated = 0
        for gen in grid.Gens if hasattr(grid, 'Gens') else []:
            p = gen.P(timestep) if hasattr(gen, 'P') and callable(getattr(gen, 'P', None)) else (getattr(gen, 'P', None) if hasattr(gen, 'P') else None)
            q = gen.Q(timestep) if hasattr(gen, 'Q') and callable(getattr(gen, 'Q', None)) else (getattr(gen, 'Q', None) if hasattr(gen, 'Q') else None)
            if p is not None:
                gen_count_updated += 1
            gen_data.append({
                'GenID': gen.ID,
                'BusID': gen.BusID,
                'P_pu': p if p is not None else 'N/A',
                'Q_pu': q if q is not None else 'N/A'
            })
        gen_df = pd.DataFrame(gen_data)
        gen_file = os.path.join(directory, f"gen_results_t{timestep}.csv")
        gen_df.to_csv(gen_file, index=False)
        print(f"保存 {gen_file}，数据行数：{len(gen_data)}，功率更新数量：{gen_count_updated}，绝对路径：{os.path.abspath(gen_file)}")

        pvw_data = []
        pvw_count_updated = 0
        for pvw in grid.PVWinds if hasattr(grid, 'PVWinds') else []:
            p_val = pvw.Pr if pvw.Pr is not None else 0.0
            q_val = pvw.Qr if pvw.Qr is not None else 0.0
            if pvw.Pr is not None:
                pvw_count_updated += 1
            pvw_data.append({
                'PVWID': pvw.ID,
                'BusID': pvw.BusID,
                'P_pu': p_val,
                'Q_pu': q_val,
                'Curtailment_Rate': pvw.CR if pvw.CR is not None else 'N/A'
            })
        pvw_df = pd.DataFrame(pvw_data)
        pvw_file = os.path.join(directory, f"pvw_results_t{timestep}.csv")
        pvw_df.to_csv(pvw_file, index=False)
        print(f"保存 {pvw_file}，数据行数：{len(pvw_data)}，功率更新数量：{pvw_count_updated}，绝对路径：{os.path.abspath(pvw_file)}")

        ess_data = []
        ess_count_updated = 0
        for ess in grid.ESSs if hasattr(grid, 'ESSs') else []:
            p_val = ess.P if ess.P is not None else 0.0
            q_val = ess.Q if ess.Q is not None else 0.0
            if ess.P is not None:
                ess_count_updated += 1
            ess_data.append({
                'ESSID': ess.ID,
                'BusID': ess.BusID,
                'P_pu': p_val,
                'Q_pu': q_val,
                'SOC': ess.SOC if ess.SOC is not None else 'N/A'
            })
        ess_df = pd.DataFrame(ess_data)
        ess_file = os.path.join(directory, f"ess_results_t{timestep}.csv")
        ess_df.to_csv(ess_file, index=False)
        print(f"保存 {ess_file}，数据行数：{len(ess_data)}，功率更新数量：{ess_count_updated}，绝对路径：{os.path.abspath(ess_file)}")

        # 保存求解状态和目标值
        summary_file = os.path.join(directory, f"summary_t{timestep}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Solver: OpenDSS\n")
            f.write(f"Result: {result}\n")
            f.write(f"Objective Value: {value:.4f}\n")
        print(f"保存 {summary_file}，绝对路径：{os.path.abspath(summary_file)}")
        return True  # 表示保存成功
    except Exception as e:
        print(f"保存 opendss 结果时发生错误 (时间步 {timestep}): {e}")
        return False  # 表示保存失败


def fix_bus_voltage_limits(grid):
    """
    修正母线的电压范围，确保 MinV 和 MaxV 不是 inf 或 -inf。
    :param grid: 配电网对象
    """
    for bus in grid.Buses if hasattr(grid, 'Buses') else []:
        if bus.MaxV == float('inf'):
            bus.MaxV = 1.5  # 设置一个合理上限，如 1.5 pu
        if bus.MinV == float('-inf'):
            bus.MinV = 0.5  # 设置一个合理下限，如 0.5 pu

def _safe_set_device_power(dev, p_value, names=("P", "_p", "Pr", "_pr", "p", "p_pu")):
    """
    兼容性写入：尽量把设备的功率写到常见的属性名上，兼容不同版本的 fpowerkit / OpenDSS wrapper。
    - dev: 设备对象（Generator / PVWind / ESS / custom）
    - p_value: 要写入的数值（通常是 pu 单位）
    - names: 依次尝试设置的属性名列表（越常用的放越前面）
    返回: (set_any, set_names)
      - set_any: bool，至少写入了一个属性则为 True
      - set_names: 写入成功的属性名列表（可能为空）
    备注：
      - 本函数不抛出异常（内部捕获），保证安全调用。
      - 若设备对象实现了 setter 方法或属性为只读，本函数会跳过并继续尝试其它名字。
    """
    set_any = False
    set_names = []
    # 如果传入的是 dict/list/ndarray，尽量把它转成标量（取当前时间步的值）：
    try:
        # 只有在 p_value 明显是可索引并且第一个元素是标量时才取第一个元素
        if not (isinstance(p_value, (int, float))):
            # 尝试把 numpy scalar / 0-d array 转成 python float
            import numpy as _np
            if isinstance(p_value, _np.ndarray) and p_value.shape == ():
                p_value = float(p_value)
    except Exception:
        # 忽略转换失败，继续使用原始 p_value
        pass

    for attr_name in names:
        try:
            # 如果对象已经有这个属性，尝试直接写
            if hasattr(dev, attr_name):
                try:
                    setattr(dev, attr_name, p_value)
                    set_any = True
                    set_names.append(attr_name)
                    # 不 break，让函数尽量写多个字段（提高兼容性）
                except Exception:
                    # 有些属性可能是只读或有验证，会抛异常，跳过
                    continue
            else:
                # 如果没有该属性，尝试创建（大多数 python 对象可动态加属性）
                try:
                    setattr(dev, attr_name, p_value)
                    set_any = True
                    set_names.append(attr_name)
                except Exception:
                    # 无法创建该属性（例如使用 __slots__ 的对象），跳过
                    continue
        except Exception:
            # 任何意外都忽略，继续尝试下一种属性名
            continue
    return set_any, set_names

def update_grid_from_model(grid, baseline_data, timestep):
    """
    从 baseline_data 字典中提取数据更新 grid 对象，并使用 _safe_set_device_power 确保兼容性。
    """

    # --- 更新 Generator ---
    gen_powers = baseline_data.get("gen_powers")
    if gen_powers is None: raise KeyError("在baseline_data中找不到 'gen_powers' 键！")
    for gen in grid.Gens:
        # 如果当前遍历到的发电机ID不在第一阶段的优化结果中
        # (这通常就是我们的虚拟发电机 gen_for_slack_bus)
        # 那么就直接跳过，不对它进行任何操作。
        if gen.ID not in gen_powers:
            continue
        p_val = gen_powers[gen.ID][timestep]
        _safe_set_device_power(gen, p_val, names=("P", "_p"))  # 使用安全写入
        _safe_set_device_power(gen, 0.0, names=("Q", "_q"))  # 假设无功为0

    # --- 更新 PVWind ---
    pvw_powers = baseline_data.get("pvw_powers")
    if pvw_powers is None: raise KeyError("在baseline_data中找不到 'pvw_powers' 键！")
    for pvw in grid.PVWinds:
        if pvw.ID not in pvw_powers: raise KeyError(f"在pvw_powers中找不到PV/Wind '{pvw.ID}' 的数据！")
        p_val = pvw_powers[pvw.ID][timestep]
        _safe_set_device_power(pvw, p_val, names=("Pr", "_pr", "P"))  # 使用安全写入
        tan_phi = (1 - pvw.PF ** 2) ** 0.5 / pvw.PF if pvw.PF != 0 else 0
        q_val = p_val * tan_phi
        _safe_set_device_power(pvw, q_val, names=("Qr", "_qr", "Q"))

    # --- 更新 ESS ---
    ess_powers = baseline_data.get("ess_powers")
    if ess_powers is None: raise KeyError("在baseline_data中找不到 'ess_powers' 键！")
    for ess in grid.ESSs:
        if ess.ID not in ess_powers: raise KeyError(f"在ess_powers中找不到ESS '{ess.ID}' 的数据！")
        p_val = ess_powers[ess.ID][timestep]
        _safe_set_device_power(ess, p_val, names=("P", "_p"))  # 使用安全写入

    # SOP/NOP 的检查逻辑保持不变
    if baseline_data.get("sop_flows") is None: raise KeyError("在baseline_data中找不到 'sop_flows' 键！")
    if baseline_data.get("nop_status") is None: raise KeyError("在baseline_data中找不到 'nop_status' 键！")



def check_grid_attributes(grid, timestep, stage="before"):
    """
    检查 grid 对象的属性值，打印调试信息。
    :param grid: 配电网对象
    :param timestep: 当前时间步
    :param stage: 检查阶段 ('before' 或 'after' OpenDSSSolver)
    """
    print(f"检查 grid 对象属性值 ({stage} OpenDSSSolver, 时间步 {timestep}):")

    # 检查 Bus 电压
    bus_updated = 0
    for bus in grid.Buses if hasattr(grid, 'Buses') else []:
        if bus.V is not None:
            bus_updated += 1
    print(f"  母线电压更新数量：{bus_updated}/{len(grid.Buses if hasattr(grid, 'Buses') else [])}")

    # 检查 Generator 功率
    gen_updated = 0
    for gen in grid.Gens if hasattr(grid, 'Gens') else []:
        p = gen.P(timestep) if hasattr(gen, 'P') and callable(getattr(gen, 'P', None)) else (getattr(gen, 'P', None) if hasattr(gen, 'P') else None)
        if p is not None:
            gen_updated += 1
    print(f"  发电机功率更新数量：{gen_updated}/{len(grid.Gens if hasattr(grid, 'Gens') else [])}")

    # 检查 PVWind 功率
    pvw_updated = 0
    for pvw in grid.PVWinds if hasattr(grid, 'PVWinds') else []:
        if pvw.Pr is not None:
            pvw_updated += 1
    print(f"  光伏/风电功率更新数量：{pvw_updated}/{len(grid.PVWinds if hasattr(grid, 'PVWinds') else [])}")

    # 检查 ESS 功率
    ess_updated = 0
    for ess in grid.ESSs if hasattr(grid, 'ESSs') else []:
        if ess.P is not None:
            ess_updated += 1
    print(f"  储能系统功率更新数量：{ess_updated}/{len(grid.ESSs if hasattr(grid, 'ESSs') else [])}")

    # 检查 Line 功率
    line_updated = 0
    for line in grid.Lines if hasattr(grid, 'Lines') else []:
        if line.P is not None:
            line_updated += 1
    print(f"  线路功率更新数量：{line_updated}/{len(grid.Lines if hasattr(grid, 'Lines') else [])}")
