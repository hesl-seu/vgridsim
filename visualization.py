import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import pandas as pd
# 设置matplotlib以正确显示中文和负号
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


import pandas as pd
import numpy as np
import os
import re


def export_simulation_data_to_excel(baseline_data, stations_list, params):
    """
    将Baseline模式下的关键数据（调度事件、各桩功率、总功率）导出到一个Excel文件。

    Args:
        baseline_data (dict): 从 solve_baseline 返回的结果字典。
        stations_list (list): 包含所有 GEVStation 对象的列表。
        params (dict): 包含仿真配置的字典。
    """
    print("开始将Baseline关键数据导出到Excel...")

    # 确保输出目录存在
    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    # 为新的综合性文件命名
    excel_path = os.path.join(output_dir, "baseline_simulation_summary.xlsx")

    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

            # 理由：遍历所有充电站，而不仅仅是第一个，以支持多站场景的数据导出。
            # --- 工作表1: 充电事件调度表 ---
            if stations_list:
                all_events_data = []
                # 遍历所有充电站
                for station in stations_list:
                    # 遍历该站的所有充电会话
                    for event in station.daily_sessions:
                        all_events_data.append({
                            "充电站ID (Station_ID)": station.station_id,
                            "充电桩ID (Spot_ID)": event.spot_id + 1,
                            "到达时间 (Arrival_Hour)": event.arrival_hour,
                            "离开时间 (Departure_Hour)": event.departure_hour,
                            "停留时长 (Duration_Hours)": event.departure_hour - event.arrival_hour
                        })

                if all_events_data:
                    # 按充电站ID、充电桩ID和到达时间排序
                    df_schedule = pd.DataFrame(all_events_data).sort_values(
                        by=["充电站ID (Station_ID)", "充电桩ID (Spot_ID)", "到达时间 (Arrival_Hour)"]
                    )
                    df_schedule.to_excel(writer, sheet_name='Charging_Event_Schedule', index=False)
                    print(f"...已将 {len(all_events_data)} 个充电事件写入工作表 'Charging_Event_Schedule'")
                else:
                    print("警告：所有充电站均无充电事件。")
            else:
                print("警告：充电站列表为空，无法生成调度事件表。")

            # --- 工作表2: 各充电桩功率曲线 (baseline_ev_spot_powers图的数据源) ---
            if "spot_powers" in baseline_data and baseline_data["spot_powers"]:
                spot_powers_pu = baseline_data["spot_powers"]
                time_steps = len(next(iter(spot_powers_pu.values())))
                start_hour = params.get('start_hour', 0)
                end_hour = params.get('end_hour', 24)
                time_axis = np.linspace(start_hour, end_hour, time_steps)
                base_power_mva = params.get('base_power', 1.0)

                df_individual_data = {'时间 (Time_h)': time_axis}
                for spot_id, powers in spot_powers_pu.items():
                    powers_kw = [p * base_power_mva * 1000 for p in powers]
                    df_individual_data[f'充电桩_{spot_id + 1}_功率 (kW)'] = powers_kw

                df_individual = pd.DataFrame(df_individual_data)
                df_individual.to_excel(writer, sheet_name='Individual_Spot_Powers', index=False)
                print(f"...已将各充电桩功率数据写入工作表 'Individual_Spot_Powers'")
            else:
                print("警告：未找到'spot_powers'数据，跳过功率曲线导出。")

            # --- 工作表2: 分离的总充电/放电功率 ---
            # 将字典转换为numpy数组便于计算
            all_powers_pu = np.array(list(spot_powers_pu.values()))  # shape: (num_spots, time_steps)

            # 计算总充电功率 (只保留正值)
            charge_powers_pu = all_powers_pu.copy()
            charge_powers_pu[charge_powers_pu < 0] = 0
            total_charge_kw = charge_powers_pu.sum(axis=0) * base_power_mva * 1000

            # 计算总放电功率 (只保留负值，然后取绝对值)
            discharge_powers_pu = all_powers_pu.copy()
            discharge_powers_pu[discharge_powers_pu > 0] = 0
            total_discharge_kw = np.abs(discharge_powers_pu.sum(axis=0)) * base_power_mva * 1000

            df_total_separated = pd.DataFrame({
                'Time (h)': time_axis,
                'Total_Charge_Power (kW)': total_charge_kw,
                'Total_Discharge_Power (kW)': total_discharge_kw
            })
            df_total_separated.to_excel(writer, sheet_name='Total_Load_Separated', index=False)
            print(f"...已将总充放电负载数据写入工作表 'Total_Load_Separated'")

        print(f"成功！Baseline EV功率数据已保存至: {os.path.abspath(excel_path)}")

    except Exception as e:
        print(f"错误：导出到Excel失败 - {e}")

def plot_voltage_snapshots(all_ts_data, seed, gui_params):
    """
    【两阶段对比版】为每个时间步生成一个独立的母线电压对比图。
    - 实线: 第一阶段近似电压 (DistFlow / RL Approx)
    - 虚线: 第二阶段精确电压 (OpenDSS)
    - 每种算法使用一种固定颜色
    """
    print("--- 正在生成【两阶段对比版】电压分时快照图 ---")

    # 1. 准备工作
    try:
        num_steps = len(next(iter(all_ts_data.values()))['step_costs'])
        if num_steps == 0: raise IndexError
    except (StopIteration, KeyError, IndexError):
        print("警告：无法确定有效的步数，跳过电压快照图绘制。")
        return

    # —— 颜色映射（家族级）+ 自动分配 ——
    algorithms = list(all_ts_data.keys())

    def _normalize_algo_name(name: str) -> str:
        s = name.strip()
        # 去掉 Two_Stage / Single_Stage / Seed 等修饰，统一为家族名
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

    # 为本次图表构建“算法名 → 颜色”的映射（未知算法自动分配）
    unknown_keys = sorted([
        _normalize_algo_name(a) for a in algorithms
        if BASE_COLOR_MAP.get(_normalize_algo_name(a)) is None and _normalize_algo_name(a) != 'BASELINE'
    ])

    algo_colors = {}
    for algo_name in algorithms:
        key = _normalize_algo_name(algo_name)
        if key == 'BASELINE':
            algo_colors[algo_name] = BASE_COLOR_MAP['BASELINE']
            continue
        color = BASE_COLOR_MAP.get(key)
        if color is None:
            idx = unknown_keys.index(key) % len(FALLBACK_PALETTE)
            color = FALLBACK_PALETTE[idx]
        algo_colors[algo_name] = color

    # 2. 创建独立的输出文件夹
    output_dir = os.path.join("results_outputs", f"voltage_snapshots_comparison_seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # 3. 遍历每一个时间步，生成一张图
    for t in range(num_steps):
        fig, ax = plt.subplots(figsize=(20, 10))

        all_bus_ids = set()
        # 收集该时间步所有算法、所有阶段的电压数据，以确定母线ID和Y轴范围
        voltages_in_step_for_ylim = []
        for algo_name, ts_data in all_ts_data.items():
            voltages_s1 = ts_data.get('voltages_data_stage1', {})
            voltages_s2 = ts_data.get('voltages_data_stage2', {})
            if voltages_s1: all_bus_ids.update(voltages_s1.keys())
            if voltages_s2: all_bus_ids.update(voltages_s2.keys())

            if voltages_s1 and t < len(next(iter(voltages_s1.values()))):
                voltages_in_step_for_ylim.extend([v_list[t] for v_list in voltages_s1.values()])
            if voltages_s2 and t < len(next(iter(voltages_s2.values()))):
                voltages_in_step_for_ylim.extend([v_list[t] for v_list in voltages_s2.values()])

        if not all_bus_ids:
            plt.close(fig);
            continue

        # 对母线ID进行自然排序
        sorted_bus_ids = sorted(list(all_bus_ids), key=lambda b: int(''.join(filter(str.isdigit, b)) or 0))

        # 4. 为每个算法绘制两条线
        for algo_name in algorithms:
            ts_data = all_ts_data[algo_name]
            color = algo_colors.get(algo_name, 'gray')  # 获取算法颜色

            # --- 绘制第一阶段 (实线) ---
            voltages_s1_dict = ts_data.get('voltages_data_stage1', {})
            if voltages_s1_dict:
                voltages_s1 = [
                    voltages_s1_dict.get(bus_id, [])[t] if t < len(voltages_s1_dict.get(bus_id, [])) else np.nan for
                    bus_id in sorted_bus_ids]
                ax.plot(sorted_bus_ids, voltages_s1, color=color, linestyle='-', marker='o', markersize=4,
                        label=f'{algo_name} (Stage 1 - Approx)')

            # --- 绘制第二阶段 (虚线) ---
            voltages_s2_dict = ts_data.get('voltages_data_stage2', {})
            if voltages_s2_dict:
                voltages_s2 = [
                    voltages_s2_dict.get(bus_id, [])[t] if t < len(voltages_s2_dict.get(bus_id, [])) else np.nan for
                    bus_id in sorted_bus_ids]
                ax.plot(sorted_bus_ids, voltages_s2, color=color, linestyle='--', marker='x', markersize=5,
                        label=f'{algo_name} (Stage 2 - OpenDSS)')

        # 5. 美化与保存
        if voltages_in_step_for_ylim:
            min_v = np.nanmin(voltages_in_step_for_ylim)
            max_v = np.nanmax(voltages_in_step_for_ylim)
            padding = (max_v - min_v) * 0.05 if (max_v - min_v) > 0.01 else 0.01
            ax.set_ylim(min_v - padding, max_v + padding)

        ax.set_xlabel('母线ID', fontsize=14)
        ax.set_ylabel('电压 (pu)', fontsize=14)
        ax.set_title(f'两阶段电压对比 (时间步: {t}, 场景 Seed: {seed})', fontsize=18)
        plt.xticks(rotation=90, fontsize=10)
        ax.legend(fontsize=12)
        ax.grid(True, axis='y', linestyle=':')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'voltage_comparison_t{t}.png')
        plt.savefig(save_path)
        plt.close(fig)

    print(f"✅ 两阶段电压对比快照图已保存至文件夹: {os.path.abspath(output_dir)}")

def plot_line_flow_snapshots(all_ts_data, seed, gui_params):
    """
    为每个时间步生成一个独立的线路潮流对比图（柱状图），并保存在新文件夹中。
    """
    print("--- 正在生成线路潮流分时快照对比图 ---")

    # 1. 准备工作
    try:
        num_steps = len(next(iter(all_ts_data.values()))['step_costs'])
        algorithms = list(all_ts_data.keys())
        if num_steps == 0: raise IndexError
    except (StopIteration, KeyError, IndexError):
        print("警告：无法确定有效的步数，跳过潮流快照图绘制。")
        return

    # 2. 创建独立的输出文件夹
    output_dir = os.path.join("results_outputs", f"line_flow_snapshots_seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # 3. 遍历每一个时间步，生成一张图
    for t in range(num_steps):
        fig, ax = plt.subplots(figsize=(18, 10))

        all_line_ids = set()
        step_data = {}
        for algo_name, ts_data in all_ts_data.items():
            flows_dict = ts_data.get('line_powers_data', {})
            if not flows_dict: continue

            current_step_flows = {line: p_list[t] for line, p_list in flows_dict.items() if t < len(p_list)}
            step_data[algo_name] = current_step_flows
            all_line_ids.update(current_step_flows.keys())

        if not all_line_ids:
            plt.close(fig)
            continue

        sorted_line_ids = sorted(list(all_line_ids), key=lambda l: int(''.join(filter(str.isdigit, l)) or 0))

        # 4. 绘制柱状图进行对比
        num_algorithms = len(algorithms)
        bar_width = 0.8 / num_algorithms
        x_indices = np.arange(len(sorted_line_ids))

        for i, algo_name in enumerate(algorithms):
            flows = [step_data.get(algo_name, {}).get(line_id, np.nan) for line_id in sorted_line_ids]
            offset = i * bar_width - (0.8 - bar_width) / 2
            ax.bar(x_indices + offset, flows, bar_width, label=algo_name)

        # 5. 美化与保存
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('有功潮流 P (pu)')
        ax.set_title(f'各算法线路有功潮流对比 (时间步: {t}, 场景 Seed: {seed})', fontsize=16)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(sorted_line_ids, rotation=90)
        ax.legend()
        ax.grid(True, axis='y', linestyle=':')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'line_flow_snapshot_t{t}.png')
        plt.savefig(save_path)
        plt.close(fig)

    print(f"✅ 线路潮流分时快照图已保存至文件夹: {os.path.abspath(output_dir)}")

def plot_line_flows(all_ts_data, seed, gui_params):
    """为所有线路有功潮流绘制时序图，并显示最大/最小潮流的包络线。
    采用与plot_voltage_profiles完全一致的健壮逻辑。
    """
    print("正在生成优化版的线路潮流时序图...")
    try:
        # 确定总步数
        num_steps = len(next(iter(all_ts_data.values()))['step_costs'])
        if num_steps == 0: raise IndexError
    except (StopIteration, KeyError, IndexError):
        print("警告：无法确定有效的步数，跳过潮流图绘制。")
        return

    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    fig, axes = plt.subplots(len(all_ts_data), 1, figsize=(15, 8 * len(all_ts_data)), sharex=True, squeeze=False)
    fig.suptitle(f'各算法线路有功潮流分布对比 (场景 Seed: {seed})', fontsize=16)

    for i, (algo_name, ts_data) in enumerate(all_ts_data.items()):
        ax = axes[i, 0]

        # 正确地获取已经统一格式的 "dict of lists"
        line_powers_dict = ts_data.get('line_powers_data', {})

        if not line_powers_dict:
            ax.text(0.5, 0.5, '无潮流数据', horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            continue

        # 将字典的值转换为Numpy数组，以便计算包络线
        # 确保所有列表长度都与num_steps一致，避免绘图错误
        all_flows_matrix = np.array([p for p in line_powers_dict.values() if len(p) == num_steps])

        # 绘制每一条线路的潮流曲线（灰色背景）
        for line_id, p_list in line_powers_dict.items():
            if len(p_list) == num_steps:
                ax.plot(time_axis, p_list, color='gray', alpha=0.3)

        # 绘制最大和最小潮流的包络线
        if all_flows_matrix.size > 0:
            min_p = np.nanmin(all_flows_matrix, axis=0)
            max_p = np.nanmax(all_flows_matrix, axis=0)
            ax.plot(time_axis, max_p, color='red', linestyle='--', label=f'最大潮流 ({np.nanmax(max_p):.3f} pu)')
            ax.plot(time_axis, min_p, color='blue', linestyle='--', label=f'最小潮流 ({np.nanmin(min_p):.3f} pu)')
            ax.fill_between(time_axis, min_p, max_p, color='lightgreen', alpha=0.3)
        # ▲▲▲▲▲ 【修正结束】 ▲▲▲▲▲

        ax.set_title(f'算法: {algo_name}')
        ax.set_ylabel('有功潮流 P (pu)')
        ax.legend()
        ax.grid(True, linestyle=':')
        ax.axhline(0, color='black', linewidth=0.5)

    axes[-1, 0].set_xlabel('时间 (小时)')
    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'line_flows_seed_{seed}.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"场景(Seed: {seed})的线路潮流图已保存至: {save_path}")
def plot_ev_spot_powers(baseline_data, gui_params):
    """
    根据Baseline的计算结果，绘制并保存EV充电桩功率变化图。

    参数:
    - baseline_data (dict): 从 solve_baseline 函数返回的详细结果字典。
    - gui_params (dict): 从GUI传入的参数字典，用于获取时间信息。
    """
    print("正在生成Baseline EV充电桩功率可视化图...")

    spot_powers_data = baseline_data.get("spot_powers")
    if not spot_powers_data:
        print("警告：在Baseline结果中未找到'spot_powers'数据，跳过绘图。")
        return

    # 从字典中提取数据
    # spot_powers_data 的格式是 {spot_id: [p1, p2, ...]}
    # 我们需要将其转换为一个 (steps, spots) 的数组
    num_spots = len(spot_powers_data)
    if num_spots == 0:
        print("警告：充电桩数量为0，跳过绘图。")
        return

    # 获取时间步数并统一所有充电桩的数据长度
    num_steps = 0
    power_lists = []
    # 确保即使某些充电桩没有数据也能处理
    for i in range(max(spot_powers_data.keys()) + 1):
        power_list = spot_powers_data.get(i, [])
        if len(power_list) > num_steps:
            num_steps = len(power_list)
        power_lists.append(power_list)

    # 填充可能不等长的数据
    for i, p_list in enumerate(power_lists):
        if len(p_list) < num_steps:
            power_lists[i] = np.pad(p_list, (0, num_steps - len(p_list)), 'constant')

    powers_array_pu = np.array(power_lists).T  # 转置为 (steps, spots)

    # 将标幺值 (pu) 转换为千瓦 (kW)
    base_power_kw = gui_params.get('base_power', 1.0) * 1000
    powers_array_kw = powers_array_pu * base_power_kw

    # 创建时间轴
    start_hour = gui_params.get('start_hour', 0)
    end_hour = gui_params.get('end_hour', 24)
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    # --- 开始绘图 ---
    # 为每个充电桩创建一个子图
    fig, axes = plt.subplots(num_spots, 1, figsize=(14, 2 * num_spots), sharex=True, squeeze=False)
    fig.suptitle('Baseline模式下各充电桩功率变化曲线', fontsize=18, y=0.95)

    for i in range(num_spots):
        ax = axes[i, 0]
        ax.plot(time_axis, powers_array_kw[:, i], label=f'充电桩 #{i + 1}')
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)  # 零功率参考线
        ax.set_ylabel('功率 (kW)')
        ax.legend(loc='upper right')
        ax.grid(True)

    axes[-1, 0].set_xlabel('时间 (小时)')

    # 保存图像
    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'baseline_ev_spot_powers.png')

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # 调整布局以适应主标题
    plt.savefig(save_path)
    plt.close()

    print(f"可视化图已成功保存至: {save_path}")


def generate_baseline_reports(baseline_data, gui_params):
    """
    根据Baseline的结果，生成分时段的电压/潮流图，
    绘制总EV负荷曲线，并保存详细数据到Excel。
    """
    print("正在生成Baseline的详细可视化报告和数据文件...")

    # --- 准备数据和目录 ---
    voltages_data = baseline_data.get("bus_voltages")
    line_flows_data = baseline_data.get("line_powers")
    spot_powers_data = baseline_data.get("spot_powers")
    output_dir_base = "results_outputs"
    os.makedirs(output_dir_base, exist_ok=True)

    if not voltages_data or not line_flows_data or not spot_powers_data:
        print("警告：结果数据不完整，无法生成详细报告。")
        return

    num_steps = len(next(iter(voltages_data.values())))
    time_axis = np.linspace(
        gui_params.get('start_hour', 0),
        gui_params.get('end_hour', 24),
        num_steps
    )
    base_power_kw = gui_params.get('base_power', 1.0) * 1000


    # --- 1. 绘制并保存每个时间步的电压分布图 ---
    voltage_plot_dir = os.path.join(output_dir_base, "baseline_voltage_profiles")
    os.makedirs(voltage_plot_dir, exist_ok=True)

    # 预处理和排序节点ID
    def get_bus_sort_key(bus_id):
        try:
            return int(''.join(filter(str.isdigit, bus_id)))
        except:
            return bus_id

    sorted_buses = sorted(voltages_data.keys(), key=get_bus_sort_key)

    for t in range(num_steps):
        fig, ax = plt.subplots(figsize=(16, 9))
        voltages_at_t = [voltages_data[bus][t] for bus in sorted_buses]

        ax.plot(sorted_buses, voltages_at_t, marker='o', linestyle='-', color='dodgerblue')
        ax.set_title(f'Baseline 电压分布 (时间步 t={t})', fontproperties="SimHei", size=18)
        ax.set_xlabel('节点 ID (排序后)', fontproperties="SimHei", size=12)
        ax.set_ylabel('电压幅值 (pu)', fontproperties="SimHei", size=12)
        ax.grid(True)
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()
        plt.savefig(os.path.join(voltage_plot_dir, f"voltage_profile_t_{t}.png"))
        plt.close(fig)
    print(f"电压分布时序图已保存至: {voltage_plot_dir}")

    # --- 2. 绘制并保存每个时间步的线路潮流图 ---
    flow_plot_dir = os.path.join(output_dir_base, "baseline_line_flows")
    os.makedirs(flow_plot_dir, exist_ok=True)

    def get_line_sort_key(line_id):
        try:
            return int(''.join(filter(str.isdigit, line_id)))
        except:
            return line_id

    sorted_lines = sorted(line_flows_data.keys(), key=get_line_sort_key)

    for t in range(num_steps):
        fig, ax = plt.subplots(figsize=(16, 9))
        flows_at_t = [line_flows_data[line][t] for line in sorted_lines]

        ax.bar(sorted_lines, flows_at_t, color='mediumseagreen')
        ax.axhline(0, color='grey', linewidth=0.8)
        ax.set_title(f'Baseline 线路有功潮流 (时间步 t={t})', fontproperties="SimHei", size=18)
        ax.set_xlabel('支路 ID (排序后)', fontproperties="SimHei", size=12)
        ax.set_ylabel('有功潮流 (pu)', fontproperties="SimHei", size=12)
        ax.grid(axis='y')
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()
        plt.savefig(os.path.join(flow_plot_dir, f"line_flow_t_{t}.png"))
        plt.close(fig)
    print(f"线路潮流时序图已保存至: {flow_plot_dir}")

    # --- 3. 将详细数据保存到Excel文件 ---
    excel_path = os.path.join(output_dir_base, 'baseline_detailed_data.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 保存电压数据
        volt_df = pd.DataFrame(voltages_data)
        volt_df = volt_df[sorted_buses]  # 按排好序的节点ID排列各列
        volt_df.index.name = "Time_Step"
        volt_df.to_excel(writer, sheet_name='Bus_Voltages')

        # 保存潮流数据
        flow_df = pd.DataFrame(line_flows_data)
        flow_df = flow_df[sorted_lines]  # 按排好序的支路ID排列各列
        flow_df.index.name = "Time_Step"
        flow_df.to_excel(writer, sheet_name='Line_Flows')

    print(f"详细的电压与潮流数据已保存至Excel文件: {excel_path}")
    #绘制并保存总充电负荷曲线
    print("...正在生成总充电负荷曲线图")
    try:
        if spot_powers_data:
            # a. 将 spot_powers_data 字典转换为 (timesteps, spots) 的Numpy数组
            num_spots = max(spot_powers_data.keys()) + 1 if spot_powers_data else 0
            powers_array_pu = np.zeros((num_steps, num_spots))
            for spot_idx, power_list in spot_powers_data.items():
                if len(power_list) == num_steps:
                    powers_array_pu[:, spot_idx] = power_list

            # b. 分别计算总充电功率（只加正值）和总放电功率（只加负值）
            total_charge_power_pu = np.sum(np.where(powers_array_pu > 0, powers_array_pu, 0), axis=1)
            total_discharge_power_pu = np.sum(np.where(powers_array_pu < 0, powers_array_pu, 0), axis=1)

            # c. 转换为kW
            total_charge_power_kw = total_charge_power_pu * base_power_kw
            total_discharge_power_kw = total_discharge_power_pu * base_power_kw

            # d. 开始绘图
            fig, ax = plt.subplots(figsize=(14, 7))

            # e. 绘制上半部分的充电曲线和填充区域
            ax.plot(time_axis, total_charge_power_kw, label='总充电功率', color='darkviolet', linewidth=2)
            ax.fill_between(time_axis, total_charge_power_kw, 0, color='darkviolet', alpha=0.3)

            # f. 绘制下半部分的放电曲线和填充区域
            ax.plot(time_axis, total_discharge_power_kw, label='总V2G放电功率', color='green', linewidth=2)
            ax.fill_between(time_axis, total_discharge_power_kw, 0, color='green', alpha=0.3)

            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            ax.set_title('Baseline 总EV负荷变化曲线 (充/放电分离)', fontproperties="SimHei", size=18)
            ax.set_xlabel('时间 (小时)', fontproperties="SimHei", size=12)
            ax.set_ylabel('总功率 (kW)', fontproperties="SimHei", size=12)
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

            # g. 保存图像
            save_path = os.path.join(output_dir_base, 'baseline_total_ev_load_separated.png')
            plt.savefig(save_path)
            plt.close(fig)
            print(f"分离式总充电负荷曲线图已保存至: {save_path}")

    except Exception as e:
        print(f"错误：生成总充电负荷曲线图时失败 - {e}")
import matplotlib.pyplot as plt
import os


def plot_spot_schedule_gantt(stations_list, target_spot_id=0):
    """
    为单个充电桩生成并保存在仿真中成功分配的充电事件的甘特图。

    Args:
        stations_list (list): 包含所有 GEVStation 对象的列表。
        target_spot_id (int): 目标充电桩的ID (索引从0开始)。
    """
    print(f"正在为 # {target_spot_id + 1} 号充电桩生成调度分配图...")

    # 在多充电站场景下，我们假设分析第一个充电站
    if not stations_list:
        print("警告：充电站列表为空，无法生成调度图。")
        return
    station = stations_list[0]

    # 1. 筛选出目标充电桩的所有充电事件
    spot_events = [s for s in station.daily_sessions if s.spot_id == target_spot_id]

    if not spot_events:
        print(f"分析完成：# {target_spot_id + 1} 号充电桩在本次仿真中没有任何充电事件。")
        return

    # 按到达时间对事件进行排序
    spot_events.sort(key=lambda x: x.arrival_hour)

    # 2. 创建图像和坐标轴
    fig, ax = plt.subplots(figsize=(16, 9))

    # 3. 绘制每一个事件为一个横条
    for i, event in enumerate(spot_events):
        start = event.arrival_hour
        duration = event.departure_hour - event.arrival_hour
        # 使用hsv颜色映射，让每个条颜色不同，更清晰
        color = plt.cm.viridis(i / len(spot_events))
        ax.barh(y=f"事件 #{i + 1}", width=duration, left=start, height=0.5, color=color, edgecolor="black")
        # 在条形内部显示时间
        ax.text(start + duration / 2, f"事件 #{i + 1}", f"{start:g}h - {event.departure_hour:g}h",
                ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # 4. 美化图表
    ax.set_xlabel("一天中的时间 (小时)", fontsize=12)
    ax.set_ylabel(f"分配给 # {target_spot_id + 1} 号充电桩的充电事件", fontsize=12)
    ax.set_title(f"充电桩 #{target_spot_id + 1} 全天任务调度甘特图", fontsize=16, fontweight='bold')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 1))
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 5. 保存图像到`results_outputs`文件夹
    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'spot_{target_spot_id + 1}_schedule.png')

    try:
        plt.savefig(output_filename)
        print(f"成功！调度图已保存为: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"错误：保存图像失败 - {e}")

    plt.close(fig)  # 关闭图像，防止在后台滞留

def plot_ess_soc(baseline_data, params):
    """
    绘制所有储能系统(ESS)的SOC时序图。
    """
    if "ess_soc" not in baseline_data or not baseline_data["ess_soc"]:
        print("信息：在baseline_data中未找到 'ess_soc' 数据，跳过ESS图像生成。")
        return

    print("正在生成ESS SOC时序图...")
    ess_soc_data = baseline_data["ess_soc"]

    # SOC数据包含T+1个时间点
    time_steps = len(next(iter(ess_soc_data.values())))
    start_hour = params.get('start_hour', 0)
    end_hour = params.get('end_hour', 24)
    # 创建一个匹配T+1个点的X轴
    time_axis = np.linspace(start_hour, end_hour, time_steps)

    plt.figure(figsize=(12, 6))
    for ess_id, soc_list in ess_soc_data.items():
        # 将SOC从pu转换为百分比
        soc_percent = [s * 100 for s in soc_list]
        plt.plot(time_axis, soc_percent, marker='.', linestyle='-', label=f"ESS #{ess_id}")

    plt.title("储能系统(ESS) SOC 时序图", fontsize=16)
    plt.xlabel("时间 (小时)")
    plt.ylabel("荷电状态 (SOC %)")
    plt.xlim(start_hour, end_hour)
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend()

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, 'ess_soc_timeseries.png')
    plt.savefig(output_filename)
    plt.close()
    print(f"成功！ESS SOC图像已保存至: {os.path.abspath(output_filename)}")


def plot_sop_flows(baseline_data, params):
    """
    绘制所有软开关(SOP)的功率潮流时序图。
    """
    if "sop_flows" not in baseline_data or not baseline_data["sop_flows"]:
        print("信息：在baseline_data中未找到 'sop_flows' 数据，跳过SOP图像生成。")
        return

    print("正在生成SOP功率时序图...")
    sop_flow_data = baseline_data["sop_flows"]
    num_sops = len(sop_flow_data)

    time_steps = len(next(iter(sop_flow_data.values()))['P1'])
    start_hour = params.get('start_hour', 0)
    end_hour = params.get('end_hour', 24)
    time_axis = np.linspace(start_hour, end_hour, time_steps)

    fig, axes = plt.subplots(num_sops, 1, figsize=(12, 4 * num_sops), sharex=True, squeeze=False)
    fig.suptitle("软开关(SOP) 功率时序图", fontsize=16, y=0.95)

    for i, (sop_id, flows) in enumerate(sop_flow_data.items()):
        ax = axes[i, 0]
        ax.plot(time_axis, flows['P1'], marker='o', linestyle='-', label='有功功率 P (pu)')
        ax.plot(time_axis, flows['Q1'], marker='x', linestyle='--', label='无功功率 Q (pu)')
        ax.set_title(f"SOP #{sop_id}")
        ax.set_ylabel("功率 (pu)")
        ax.grid(True)
        ax.legend()

    plt.xlabel("时间 (小时)")

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, 'sop_flows_timeseries.png')
    plt.savefig(output_filename)
    plt.close()
    print(f"成功！SOP功率图像已保存至: {os.path.abspath(output_filename)}")


def plot_nop_status(baseline_data, params):
    """
    绘制所有常开点(NOP)的开断状态时序图。
    """
    if "nop_status" not in baseline_data or not baseline_data["nop_status"]:
        print("信息：在baseline_data中未找到 'nop_status' 数据，跳过NOP图像生成。")
        return

    print("正在生成NOP开断状态时序图...")
    nop_status_data = baseline_data["nop_status"]

    time_steps = len(next(iter(nop_status_data.values())))
    start_hour = params.get('start_hour', 0)
    end_hour = params.get('end_hour', 24)
    time_axis = np.linspace(start_hour, end_hour, time_steps)

    plt.figure(figsize=(12, 6))
    for nop_id, status_list in nop_status_data.items():
        # 使用阶梯图(step plot)可以更清晰地表示状态切换
        plt.step(time_axis, status_list, where='post', label=f"NOP #{nop_id}")

    plt.title("常开点(NOP) 开断状态时序图", fontsize=16)
    plt.xlabel("时间 (小时)")
    plt.ylabel("状态")
    plt.yticks([0, 1], ["关 (Off)", "开 (On)"])
    plt.ylim(-0.1, 1.1)
    plt.xlim(start_hour, end_hour)
    plt.grid(True)
    plt.legend()

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, 'nop_status_timeseries.png')
    plt.savefig(output_filename)
    plt.close()
    print(f"成功！NOP状态图像已保存至: {os.path.abspath(output_filename)}")

def plot_line_flow_snapshots_comparison(all_ts_data, seed, gui_params):
    """
    为每个时间步生成分阶段、多算法的线路潮流对比图。
    - 每个时刻一张图，包含上下两个子图，分别代表第一和第二阶段。
    - 每个子图内，用分组柱状图对比各算法的线路潮流。
    """
    print("--- 正在生成分阶段、多算法线路潮流对比快照图 ---")

    # 1. 准备工作
    try:
        num_steps = len(next(iter(all_ts_data.values()))['step_costs'])
        algorithms = list(all_ts_data.keys())
        if num_steps == 0: raise IndexError
    except (StopIteration, KeyError, IndexError):
        print("警告：无法确定有效的步数，跳过潮流对比快照图绘制。")
        return

    # 2. 创建独立的输出文件夹
    output_dir = os.path.join("results_outputs", f"line_flow_snapshots_comparison_seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # 3. 遍历每一个时间步
    for t in range(num_steps):
        fig, axes = plt.subplots(2, 1, figsize=(20, 16), sharex=True)
        fig.suptitle(f'线路有功潮流对比 (时间步: {t}, 场景 Seed: {seed})', fontsize=20)

        # --- 数据收集 ---
        stage1_data, stage2_data = {}, {}
        all_line_ids = set()

        for algo_name, ts_data in all_ts_data.items():
            # 提取第一阶段数据
            s1_flows = ts_data.get('line_powers_data_stage1', {})
            if s1_flows:
                all_line_ids.update(s1_flows.keys())
                stage1_data[algo_name] = {line: p_list[t] for line, p_list in s1_flows.items() if t < len(p_list)}

            # 提取第二阶段数据
            s2_flows = ts_data.get('line_powers_data_stage2', {})
            if s2_flows:
                all_line_ids.update(s2_flows.keys())
                stage2_data[algo_name] = {line: p_list[t] for line, p_list in s2_flows.items() if t < len(p_list)}

        if not all_line_ids:
            plt.close(fig)
            continue

        sorted_line_ids = sorted(list(all_line_ids), key=lambda l: int(''.join(filter(str.isdigit, l)) or 0))

        # --- 绘图参数 ---
        num_algorithms = len(algorithms)
        bar_width = 0.8 / num_algorithms
        x_indices = np.arange(len(sorted_line_ids))

        # --- 绘制子图 ---
        plot_titles = ['第一阶段 (Linear DistFlow Approximation)', '第二阶段 (OpenDSS Precise Calculation)']
        data_sources = [stage1_data, stage2_data]

        for i, ax in enumerate(axes):
            for j, algo_name in enumerate(algorithms):
                source = data_sources[i]
                flows = [source.get(algo_name, {}).get(line_id, 0) for line_id in sorted_line_ids]
                offset = j * bar_width - (0.8 - bar_width) / 2
                ax.bar(x_indices + offset, flows, bar_width, label=algo_name)

            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_title(plot_titles[i], fontsize=16)
            ax.set_ylabel('有功潮流 P (pu)', fontsize=14)
            ax.legend()
            ax.grid(True, axis='y', linestyle=':')

        plt.xticks(x_indices, sorted_line_ids, rotation=90)
        plt.xlabel('支路 ID', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # --- 保存图像 ---
        save_path = os.path.join(output_dir, f'line_flow_comparison_t{t}.png')
        plt.savefig(save_path)
        plt.close(fig)

    print(f"✅ 分阶段潮流对比快照图已保存至文件夹: {os.path.abspath(output_dir)}")

