"""
该模块负责创建和配置仿真所需的电网（Grid）对象。

核心功能:
1.  提供一个主函数 `create_grid` 作为统一入口，用于生成一个完整的、包含所有组件的电网模型。
2.  从 fpowerkit 库加载标准的 IEEE 测试案例（如 IEEE 33, IEEE 69）作为基础拓扑。
3.  从外部数据文件 `data/grid_parameters.xlsx` 中读取详细的、自定义的组件参数，
    包括负荷曲线、电价、发电机、分布式能源（光伏、风电、储能）、以及软开关（SOP/NOP）。
4.  将从数据文件中读取的参数加载到基础电网模型中，覆盖或添加相应的组件。
5.  包含健壮性修复和参数标准化逻辑，确保生成的电网对象在不同环境下行为一致。
"""

import os
import pandas as pd
import sys
import numpy as np
from fpowerkit import Grid, Bus, Line, Generator, PVWind, ESS
from fpowerkit.cases import PDNCases
from sop_nop import SOP, NOP
from config import CORE_PARAMS, PATHS
from feasytools.tfunc import SegFunc, ConstFunc

# 定义项目内的关键路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#GRID_PARAMS_FILE = os.path.join(BASE_DIR, 'data', 'grid_parameters.xlsx')


def generate_stochastic_power_profile(predicted_profile, error_level=0.08):
    """
    根据预测功率曲线和误差水平，生成一个随机的实际功率曲线。
    这是为了模拟可再生能源发电的预测不确定性。

    参数:
    - predicted_profile (list or np.array): 预测的功率值列表 (来自Excel插值后)。
    - error_level (float): 不确定性水平，用作标准差与均值的比例。

    返回:
    - np.array: 带有随机扰动的实际功率曲线。
    """
    stochastic_profile = []
    for p_predicted in predicted_profile:
        # 只有当预测值大于0时才施加扰动，避免在夜间（无光/风时）产生异常功率
        if p_predicted > 0:
            # 均值(mu)就是预测值
            mu = p_predicted
            # 标准差(sigma)是预测值的一个百分比
            sigma = p_predicted * error_level

            # 从正态(高斯)分布中采样一个随机值作为实际出力
            p_actual = np.random.normal(loc=mu, scale=sigma)

            # 对结果进行约束，确保其符合物理现实：
            # 1. 实际出力不能为负值。
            # 2. 为简化模型，假设实际出力不会超过当下的理论预测值。
            p_actual = np.clip(p_actual, 0, p_predicted)

            stochastic_profile.append(p_actual)
        else:
            # 如果预测值为0，则实际值也为0
            stochastic_profile.append(0)

    return np.array(stochastic_profile)


def load_generators_from_excel(grid):
    """
    从Excel文件的'Generators'工作表中加载常规发电机(Gen)的配置。
    一个关键特性是：如果成功读取到数据，它会先清除案例自带的默认发电机，
    然后再添加Excel中定义的发电机，实现了完全的自定义配置。
    """
    try:
        df_gens = pd.read_excel(PATHS["grid_params_excel"], sheet_name="Generators")

        # 只要能成功读取到数据，就先清除案例自带的默认发电机
        if not df_gens.empty:
            #print("--- 诊断信息：检测到自定义发电机数据，正在清除案例自带的默认发电机... ---")
            # 使用列表推导式来安全地获取并移除所有现有的发电机
            for gen_id in [g.ID for g in grid.Gens]:
                grid.DelGen(gen_id)
            #print(f"--- 诊断信息：已清除所有默认发电机。---")

        # 遍历Excel中的每一行，创建并添加新的发电机
        for _, row in df_gens.iterrows():
            # 确保所有必需的列都存在，以进行健壮的数据加载
            required_cols = ["ID", "BusID", "Pmax_pu", "Pmin_pu", "Qmax_pu", "Qmin_pu", "CostA", "CostB", "CostC",
                             "RealisticPmax_pu"]
            if not all(col in row for col in required_cols):
                print(f"警告: Generators工作表中缺少必要列，跳过行: {row.to_dict()}")
                continue

            #print(f"--- 诊断信息：正在从Excel加载发电机 '{row['ID']}' 并放置到母线 '{row['BusID']}'... ---")

            gen = Generator(
                id=str(row["ID"]),
                busid=str(row["BusID"]),
                pmax_pu=row["Pmax_pu"],
                pmin_pu=row["Pmin_pu"],
                qmax_pu=row["Qmax_pu"],
                qmin_pu=row["Qmin_pu"],
                costA=row["CostA"],
                costB=row["CostB"],
                costC=row["CostC"]
            )
            # 附加在优化模型中使用的自定义“真实物理上限”属性
            gen.RealisticPmax = row["RealisticPmax_pu"]

            # 将从Excel创建的发电机添加到电网中
            grid.AddGen(gen)

        #print("--- 诊断信息：已成功从Excel加载并配置所有自定义发电机。 ---")

    except Exception as e:
        # 如果工作表不存在或读取失败，则打印警告，并继续使用案例自带的默认发电机
        print(f"警告：加载自定义发电机失败 - {e}。将使用案例自带的默认发电机。")

    return grid


def load_electricity_price(gui_params):
    """
    从Excel加载24小时的电价数据，并根据GUI传入的仿真参数（起止时间、步长）
    对电价曲线进行线性插值，以匹配仿真时间轴。
    """
    if gui_params is None:
        return [0.05] * 24  # 如果没有提供参数，返回一个默认值

    # 从GUI参数字典中获取时间设置
    start_hour = gui_params['start_hour']
    end_hour = gui_params['end_hour']
    step_minutes = gui_params['step_minutes']

    try:
        df = pd.read_excel(PATHS["grid_params_excel"], sheet_name="ElectricityPrice")
        # 从Excel中读取24个整点时刻的电价
        hourly_prices = [
            df[f"Price_t{t}"].iloc[0] if f"Price_t{t}" in df.columns and pd.notna(df[f"Price_t{t}"].iloc[0]) else 0.0
            for t in range(24)]

        # 创建原始的时间点（0, 1, 2, ..., 23）
        original_time_points = np.arange(24)
        # 根据GUI设置创建新的、更高分辨率的时间点
        step_duration_hours = step_minutes / 60.0
        new_time_points = np.arange(start_hour, end_hour, step_duration_hours)

        # 使用NumPy的interp函数进行线性插值
        interpolated_prices = np.interp(new_time_points, original_time_points, hourly_prices)

        # 如果Excel中有-1等无效值，用前一个有效值填充
        last_valid_price = 0.05
        for i, price in enumerate(hourly_prices):
            if price != -1.0:
                last_valid_price = price
            else:
                hourly_prices[i] = last_valid_price

        # 使用填充后的数据再次插值，确保结果的有效性
        interpolated_prices = np.interp(new_time_points, original_time_points, hourly_prices, left=hourly_prices[0],
                                        right=hourly_prices[-1])

        print(f"电价数据已重新采样，总步数: {len(interpolated_prices)}")
        return list(interpolated_prices)
    except Exception as e:
        print(f"错误: 加载电价失败 - {e}")
        # 如果加载失败，返回一个默认电价列表以保证程序能继续运行
        time_steps = int((end_hour - start_hour) * (60 / step_minutes))
        return [0.05] * time_steps


def load_bus_loads(grid, gui_params):
    """
    从Excel加载母线负荷。此函数能根据GUI选择的电网模型（如'ieee33'）
    动态地选择要读取的Sheet页（如'BusLoads_ieee33'），实现了数据与模型的自动匹配。
    """
    #print("--- 诊断信息：正在尝试从 Excel 文件加载负荷... ---")

    grid_model_name = gui_params['grid_model']
    # 根据模型名称动态构建Sheet页的名称
    sheet_name_to_load = f"BusLoads_{grid_model_name}"

    start_hour = gui_params['start_hour']
    end_hour = gui_params['end_hour']
    step_minutes = gui_params['step_minutes']

    try:
        # 使用动态构建的Sheet页名称来读取数据
        df = pd.read_excel(PATHS["grid_params_excel"], sheet_name=sheet_name_to_load)
        #print(f"--- 诊断信息：成功从 '{GRID_PARAMS_FILE}' 的 '{sheet_name_to_load}' Sheet页读取了 {len(df)} 行数据。---")
    except Exception as e:
        print(f"--- 致命错误：加载母线负荷失败！未能读取 '{sheet_name_to_load}' Sheet页。---")
        print(f"--- 错误信息是: {e} ---")
        return grid

    # 为了避免插值的时间轴错位，总是先生成一个完整的24小时高分辨率时间轴
    full_day_steps = int(24 * (60 / step_minutes))
    full_day_time_axis = np.linspace(0, 24, full_day_steps, endpoint=False)
    original_time_points = np.arange(24)

    # 计算出仿真窗口在完整时间轴上的起止索引
    start_step_index = int((start_hour) * (60 / step_minutes))
    end_step_index = int((end_hour) * (60 / step_minutes))

    timestep_seconds = step_minutes * 60
    overwrite_success = False

    # 遍历Excel中的每一行（代表一个母线）
    for _, row in df.iterrows():
        bus_id = str(row["BusID"])
        # 如果Excel中的母线ID不存在于当前电网模型中，则跳过
        if bus_id not in grid.BusNames:
            continue

        bus = grid.Bus(bus_id)
        if bus:
            # --- 处理有功负荷 Pd ---
            hourly_pd = [row[f"Pd_t{t}"] if f"Pd_t{t}" in row and pd.notna(row[f"Pd_t{t}"]) else 0.0 for t in range(24)]
            full_day_interpolated_pd = np.interp(full_day_time_axis, original_time_points, hourly_pd)
            full_day_interpolated_pd = generate_stochastic_power_profile(full_day_interpolated_pd,
                                                                         error_level=0.05)  # 假设5%的误差
            # 从完整的插值结果中，切出仿真窗口所需的部分
            final_pd_slice = full_day_interpolated_pd[start_step_index:end_step_index]
            # 使用 fpowerkit 的 SegFunc (分段函数) 来表示这个时序负荷
            time_points_sec = [(i * timestep_seconds, val) for i, val in enumerate(final_pd_slice)]
            bus.Pd = SegFunc(time_points_sec, list(final_pd_slice))

            # --- 处理无功负荷 Qd (逻辑同上) ---
            hourly_qd = [row[f"Qd_t{t}"] if f"Qd_t{t}" in row and pd.notna(row[f"Qd_t{t}"]) else 0.0 for t in range(24)]
            full_day_interpolated_qd = np.interp(full_day_time_axis, original_time_points, hourly_qd)
            full_day_interpolated_qd = generate_stochastic_power_profile(full_day_interpolated_qd,
                                                                         error_level=0.05)  # 假设5%的误差
            final_qd_slice = full_day_interpolated_qd[start_step_index:end_step_index]
            time_points_sec_q = [(i * timestep_seconds, val) for i, val in enumerate(final_qd_slice)]
            bus.Qd = SegFunc(time_points_sec_q, list(final_qd_slice))

            overwrite_success = True

    if overwrite_success:
        #print("--- 诊断信息：已成功将Excel中的负荷数据覆盖到电网模型。---")
        pass

    return grid


def load_distributed_energy(grid, gui_params):
    """
    从Excel加载所有分布式能源（DER），包括光伏(PV)、风电(Wind)和储能(ESS)。
    光伏和风电的出力曲线会经过插值和随机化处理。
    """
    start_hour = gui_params['start_hour']
    end_hour = gui_params['end_hour']
    step_minutes = gui_params['step_minutes']

    # ------------------- 1. 加载光伏 (PV) 和风电 (Wind) -------------------
    if CORE_PARAMS["distributed_energy"]["pv"] or CORE_PARAMS["distributed_energy"]["wind"]:
        try:
            df_pvw = pd.read_excel(PATHS["grid_params_excel"], sheet_name="PVWind")
            original_time_points = np.arange(24)
            step_duration_hours = step_minutes / 60.0
            new_time_points = np.arange(start_hour, end_hour, step_duration_hours)
            timestep_seconds = step_minutes * 60
            for _, row in df_pvw.iterrows():
                pvw_id = str(row["ID"])
                bus_id = str(row["BusID"])
                pvw_type = str(row["Type"]).lower()

                if (pvw_type == "pv" and CORE_PARAMS["distributed_energy"]["pv"]) or \
                        (pvw_type == "wind" and CORE_PARAMS["distributed_energy"]["wind"]):
                    # 先检查母线在不在当前网架里
                    if hasattr(grid, "_bnames") and bus_id not in grid._bnames:
                        print(f"警告：PV/Wind {pvw_id} 的母线 {bus_id} 未在当前配电网模型中找到，跳过此机组。")
                        continue
                    # 从Excel读取24小时的预测出力
                    p_values = [row[f"P_t{t}"] if f"P_t{t}" in row and pd.notna(row[f"P_t{t}"]) else 0.0 for t in
                                range(24)]

                    # 首先进行插值，得到与仿真步长匹配的确定性预测曲线
                    predicted_p_profile = np.interp(new_time_points, original_time_points, p_values)

                    # 然后，调用辅助函数为预测曲线添加随机扰动，模拟预测误差
                    UNCERTAINTY_LEVEL = 0.08
                    actual_p_profile = generate_stochastic_power_profile(predicted_p_profile, UNCERTAINTY_LEVEL)

                    # 最后，使用这个带有随机性的实际出力曲线来创建 fpowerkit 的分段函数
                    time_points_sec = [(i * timestep_seconds, val) for i, val in enumerate(actual_p_profile)]
                    p_func = SegFunc(time_points_sec, list(actual_p_profile))

                    pvw = PVWind(
                        pvw_id,
                        bus_id,
                        p=p_func,  # 将带有不确定性的功率函数赋给对象
                        pf=row["PF"] if pd.notna(row["PF"]) else 0.95,
                        cc=row["CC"] if pd.notna(row["CC"]) else 1.5,
                        tag=pvw_type
                    )
                    grid.AddPVWind(pvw)
        except ValueError as e:
            print(f"警告：读取Excel文件 'PVWind' 工作表时发生错误 - {e}，跳过光伏/风电数据加载。")

    # ------------------- 2. 加载储能系统 (ESS) -------------------
    # 储能系统的参数是静态的，不是时间序列，因此直接读取即可。
    if CORE_PARAMS["distributed_energy"]["ess"]:
        try:
            df_ess = pd.read_excel(PATHS["grid_params_excel"], sheet_name="ESS")
            for _, row in df_ess.iterrows():
                bus_id = str(row["BusID"])

                # 检查母线是否存在
                if hasattr(grid, "_bnames") and bus_id not in grid._bnames:
                    print(f"警告：ESS {row['ID']} 的母线 {bus_id} 未在当前配电网模型中找到，跳过此储能。")
                    continue
                # 1. 获取确定性的初始值和容量
                init_soc_deterministic = row["Init_Elec_puh"] if pd.notna(row["Init_Elec_puh"]) else 0.25
                ess_cap = row["Cap_puh"] if pd.notna(row["Cap_puh"]) else 0.5

                # 2. 生成随机扰动 (例如：均值为0，标准差为0.2，即20%的扰动)
                soc_noise_factor = np.random.normal(loc=1.0, scale=0.2)

                # 3. 应用扰动并确保SOC在 [0, 容量] 范围内
                init_soc_stochastic = init_soc_deterministic * soc_noise_factor
                init_soc_stochastic = np.clip(init_soc_stochastic, 0, ess_cap)
                ess = ESS(
                    str(row["ID"]),
                    bus_id,
                    cap_puh=row["Cap_puh"] if pd.notna(row["Cap_puh"]) else 0.5,
                    ec=row["EC"] if pd.notna(row["EC"]) else 0.9,
                    ed=row["ED"] if pd.notna(row["ED"]) else 0.9,
                    pc_max=row["Pc_max"] if pd.notna(row["Pc_max"]) else 0.1,
                    pd_max=row["Pd_max"] if pd.notna(row["Pd_max"]) else 0.1,
                    pf=row["PF"] if pd.notna(row["PF"]) else 0.95,
                    # policy设为None表示其充放电行为由外部优化器（Baseline或RL）决定
                    policy=None,
                    cprice=None,
                    dprice=None,
                    init_elec_puh=init_soc_stochastic # <-- 使用随机化的值
                )
                grid.AddESS(ess)
        except ValueError as e:
            print(f"警告：读取Excel文件 'ESS' 工作表时发生错误 - {e}，跳过储能数据加载。")

    return grid


def load_sop_nop(grid):
    """
    从Excel文件加载软开关(SOP)和常开点(NOP)的参数。
    在加载时会验证SOP/NOP所连接的母线是否存在于当前电网模型中。
    """
    # 加载SOP
    if CORE_PARAMS["sop_nodes_active"]:
        try:
            df_sop = pd.read_excel(PATHS["grid_params_excel"], sheet_name="SOP")
            if not hasattr(grid, 'SOPs'):
                grid.SOPs = {}
            for _, row in df_sop.iterrows():
                sop_id = str(row["ID"])
                bus1 = str(row["Bus1"])
                bus2 = str(row["Bus2"])
                # 验证母线ID是否存在于当前配电网模型中
                if grid.Bus(bus1) is None or grid.Bus(bus2) is None:
                    print(f"警告：SOP {sop_id} 的母线 {bus1} 或 {bus2} 未在配电网模型中找到，跳过此设备。")
                    continue
                sop = SOP(
                    sop_id,
                    bus1,
                    bus2,
                    p_max_pu=row["P_max_pu"] if pd.notna(row["P_max_pu"]) else 0.5,
                    q_max_pu=row["Q_max_pu"] if pd.notna(row["Q_max_pu"]) else 0.3,
                    loss_coeff=row["Loss_Coeff"] if pd.notna(row["Loss_Coeff"]) else 0.05,
                    active=True
                )
                grid.SOPs[sop.ID] = sop
        except ValueError as e:
            print(f"警告：读取Excel文件 'SOP' 工作表时发生错误 - {e}，跳过SOP数据加载。")

    # 加载NOP
    if CORE_PARAMS["nop_nodes_active"]:
        try:
            df_nop = pd.read_excel(PATHS["grid_params_excel"], sheet_name="NOP")
            if not hasattr(grid, 'NOPs'):
                grid.NOPs = {}
            for _, row in df_nop.iterrows():
                nop_id = str(row["ID"])
                bus1 = str(row["Bus1"])
                bus2 = str(row["Bus2"])
                # 同样进行母线ID验证
                if grid.Bus(bus1) is None or grid.Bus(bus2) is None:
                    print(f"警告：NOP {nop_id} 的母线 {bus1} 或 {bus2} 未在配电网模型中找到，跳过此设备。")
                    continue
                nop = NOP(
                    nop_id,
                    bus1,
                    bus2,
                    r_pu=row["R_pu"] if pd.notna(row["R_pu"]) else 0.001,
                    x_pu=row["X_pu"] if pd.notna(row["X_pu"]) else 0.01,
                    max_I_kA=row["Max_I_kA"] if pd.notna(row["Max_I_kA"]) else 0.9,
                    active=False  # NOP的状态由优化模型决定，此处仅为初始化
                )
                grid.NOPs[nop.ID] = nop
        except ValueError as e:
            print(f"警告：读取Excel文件 'NOP' 工作表时发生错误 - {e}，跳过NOP数据加载。")
    return grid


def load_station_info():
    """从Excel文件的'EVStation'工作表中加载所有充电站的信息。"""
    try:
        df_ev = pd.read_excel(PATHS["grid_params_excel"], sheet_name="EVStation")
        # 将DataFrame转换为一个字典列表，每行一个字典，方便使用
        stations_info = df_ev.to_dict('records')
        print(f"成功加载 {len(stations_info)} 个充电站的信息。")
        return stations_info
    except Exception as e:
        print(f"警告：读取Excel文件 'EVStation' 工作表时发生错误 - {e}，返回空列表。")
        return []


def create_ieee_grid(model="ieee33"):
    """根据指定的模型名称（如'ieee33'），从fpowerkit中创建一个基础IEEE配电网案例。"""
    if model == "ieee33":
        return PDNCases.IEEE33()
    elif model == "ieee69":
        return PDNCases.IEEE69()
    elif model == "ieee123":
        return PDNCases.IEEE123()
    else:
        print(f"错误：不支持的模型 {model}，将默认使用 IEEE 33。")
        return PDNCases.IEEE33()


def create_grid(model="ieee33", gui_params=None):
    """
    (健壮性增强版) 创建完整配电网模型的主函数。

    这是一个工厂函数，它按顺序执行以下操作：
    1. 创建一个基础的IEEE电网拓扑。
    2. 手动设置和修正关键的电网基准参数（SB, UB），以保证兼容性。
    3. 调用所有其他的 `load_*` 函数，将Excel中的自定义数据加载并应用到电网对象上。
    4. 对所有组件进行最终的参数标准化和健全性检查。
    """
    # 1. 创建一个基础电网案例
    grid = create_ieee_grid(model)

    # 为了应对不同版本的 fpowerkit 库可能存在的差异，在此处手动检查并设置 SB 和 UB 属性。
    # 这确保了从本函数返回的 grid 对象一定包含我们需要的基准功率和基准电压值。
    if model == "ieee33":
        grid.SB = 1.0  # IEEE 33 节点的基准功率是 1.0 MVA
        grid.UB = 12.66  # IEEE 33 节点的基准电压是 12.66 kV
    elif model == "ieee69":
        grid.SB = 10.0  # IEEE 69 节点的基准功率是 10.0 MVA
        grid.UB = 12.66  # IEEE 69 节点的基准电压是 12.66 kV
    elif model == "ieee123":
        grid.SB = 5.0  # IEEE 69 节点的基准功率是 5.0 MVA
        grid.UB = 4.16  # IEEE 69 节点的基准电压是 4.16 kV
    else:
        # 为其他可能的模型设置一个默认值，以防万一
        if not hasattr(grid, 'SB'): grid.SB = 1.0
        if not hasattr(grid, 'UB'): grid.UB = 12.66
    print(f"已确保电网模型 '{model}' 的基准功率 SB={grid.SB} MVA, 基准电压 UB={grid.UB} kV ---")

    # 2. 为所有母线设置一个默认的电压约束范围
    for bus in grid.Buses:
        bus.MinV = 0.95
        bus.MaxV = 1.05

    # 3. 如果提供了GUI参数（在evaluate_agents.py当中，GUI参数已经被config.py定义了），则调用所有数据加载函数
    if gui_params:
        grid = load_bus_loads(grid, gui_params)
        grid = load_distributed_energy(grid, gui_params)
        grid = load_sop_nop(grid)
        grid = load_generators_from_excel(grid)

    # 4. 对所有发电机进行最终的参数标准化
    # 确保每个发电机都有优化模型所需的成本函数和真实上限属性
    REALISTIC_PMAX_PU = 5
    TARGET_COST_A = 0.1
    TARGET_COST_B = 600
    TARGET_COST_C = 10
    for gen in grid.Gens:
        if not hasattr(gen, 'RealisticPmax'):
            gen.RealisticPmax = REALISTIC_PMAX_PU
        if gen.CostA is None: gen.CostA = ConstFunc(TARGET_COST_A)
        if gen.CostB is None: gen.CostB = ConstFunc(TARGET_COST_B)
        if gen.CostC is None: gen.CostC = ConstFunc(TARGET_COST_C)
        print(
            f"最终发电机配置 {gen.ID} (在母线 {gen.BusID}): RealisticPmax={getattr(gen, 'RealisticPmax', 'N/A')}, CostB={gen.CostB(0)}")

    # 5. [模型处理] 为每个NOP在电网中添加一条对应的、默认不激活的线路。
    # 这是为了在优化模型中通过控制这条线路的激活状态来模拟NOP的开断。
    if hasattr(grid, 'NOPs'):
        for nop_id, nop in grid.NOPs.items():
            nop_line = Line(
                id=f"line_for_{nop_id}",
                fbus=nop.Bus1,
                tbus=nop.Bus2,
                r_pu=nop.R,
                x_pu=nop.X,
                active=False  # 默认不激活
            )
            grid.AddLine(nop_line)
        #print(f"--- 诊断信息：已将 {len(grid.NOPs)} 个 NOP 模拟为默认不激活的线路。---")

    return grid