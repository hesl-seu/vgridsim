import numpy as np
import random
import os
from dataclasses import dataclass, field
import pandas as pd
@dataclass
class EVParameters:
    """定义电动汽车的通用物理参数"""
    capacity_kwh: float = 70.0
    max_charge_kw: float = 60.0
    max_discharge_kw: float = 25.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.9


@dataclass
class EVChargeSession:
    """定义单次EV充电会话的场景信息"""
    ev_id: int
    spot_id: int
    arrival_hour: int
    departure_hour: int
    initial_soc: float
    final_soc: float = field(init=False, default=0.0)


# --- G-EVs 充电站核心类 ---
class GEVStation:
    """
    G-EVs充电站模块。
    此版本重点修复了充电桩分配逻辑，确保时空唯一性。
    """
    def __init__(self, station_id: str, num_spots: int = 30, ev_params: EVParameters = None):
        self.station_id = station_id
        self.num_spots = num_spots
        self.ev_params = ev_params if ev_params else EVParameters()
        self.daily_sessions: list[EVChargeSession] = []
        # 日志中加入ID
        print(f"G-EVs充电站 (ID: {self.station_id}) 已创建，包含 {self.num_spots} 个充电桩。")
    def _is_timeslot_free(self, schedule: list, new_arrival: int, new_departure: int) -> bool:
        """
        内部辅助函数：检查新的时间段是否与已安排的时间段重叠。
        """
        for arrival, departure in schedule:
            # 检查重叠条件: (StartA < EndB) and (EndA > StartB)
            if new_arrival < departure and new_departure > arrival:
                return False  # 如果有重叠，则不空闲
        return True  # 如果循环结束都没有重叠，则空闲

    def load_scenarios_from_csv(self, all_scenarios_df: pd.DataFrame):
        """
        从一个包含所有充电站场景的DataFrame中，加载属于本站的充电会话。

        参数:
            all_scenarios_df (pd.DataFrame): 包含所有充电事件的完整DataFrame。
        """
        self.daily_sessions = []

        # 筛选出只属于当前充电站实例的场景数据
        # 假设 GEVStation 实例有一个 station_id 属性
        station_scenarios_df = all_scenarios_df[all_scenarios_df['station_id'] == self.station_id]

        if station_scenarios_df.empty:
            print(f"【信息】: 在CSV数据中，未找到任何属于充电站 '{self.station_id}' 的充电事件。")
            return

        # 验证充电桩ID是否超出本站范围
        if 'spot_id_in_station' in station_scenarios_df.columns:
            max_spot_id = station_scenarios_df['spot_id_in_station'].max()
            if max_spot_id >= self.num_spots:
                print(f"【错误】: 数据文件中有分配给充电站 '{self.station_id}' 的充电桩ID ({max_spot_id})")
                print(f"         超出了该站实际拥有的充电桩数量 ({self.num_spots})。请检查您的数据！")
                raise ValueError(f"充电站 {self.station_id} 的充电桩ID越界。")

        # 将DataFrame中的每一行转换为EVChargeSession对象
        for index, row in station_scenarios_df.iterrows():
            session = EVChargeSession(
                ev_id=index,
                spot_id=row['spot_id_in_station'],
                arrival_hour=int(row['arrival_hour']),
                departure_hour=int(row['departure_hour']),
                initial_soc=float(row['initial_soc'])
            )
            self.daily_sessions.append(session)

        print(f"成功为充电站 '{self.station_id}' 从CSV加载了 {len(self.daily_sessions)} 个充电会话。")



    def generate_daily_scenarios(self, num_evs_to_generate: int = 120):
        """
        生成一整天的随机EV充电会话场景。
        采用三高峰（早、午、晚）混合正态分布模型生成到达时间。
        晚高峰车辆最多。
        """
        self.daily_sessions = []
        spot_schedules = [[] for _ in range(self.num_spots)]

        # --- 1. 定义三高峰时段参数 ---
        # 格式为: (中心时间, 时间标准差, 车辆数权重)
        # 注意：所有权重加起来必须等于 1.0
        PEAK_MORNING = (8, 1.5, 0.30)  # 早高峰: 8点为中心, 标准差1.5小时, 占总车流30%
        PEAK_NOON = (12, 1.0, 0.20)  # 午高峰: 12点为中心, 标准差1小时, 占总车流20%
        PEAK_EVENING = (18, 2.0, 0.50)  # 晚高峰: 18点为中心, 标准差2小时, 占总车流50% (权重最高)

        peaks = [PEAK_MORNING, PEAK_NOON, PEAK_EVENING]
        peak_weights = [p[2] for p in peaks]


        generated_count = 0
        attempts = 0
        while generated_count < num_evs_to_generate and attempts < num_evs_to_generate * 5:
            attempts += 1

            # --- 2. 生成随机到达时间 ---
            # 2.1. 首先根据权重随机选择一个高峰
            chosen_peak = random.choices(peaks, weights=peak_weights, k=1)[0]

            # 2.2. 从选中的高峰对应的正态分布中生成一个具体的到达时间
            arrival_float = np.random.normal(loc=chosen_peak[0], scale=chosen_peak[1])

            # 2.3. 将浮点数时间转换为整数小时，并确保在0-23点的范围内
            arrival = int(round(arrival_float))
            arrival = max(0, min(23, arrival))

            # --- 3. 生成整数停留时间 (快充模式) ---
            # 随机生成1或2小时的整数停留时间
            stay_duration = random.randint(1, 3)

            departure = arrival + stay_duration
            departure = min(24, departure)

            if arrival >= departure:
                continue

            # --- 4. 分配充电桩 (逻辑不变) ---
            available_spots = list(range(self.num_spots))
            random.shuffle(available_spots)

            assigned_spot = -1
            for spot_id in available_spots:
                if self._is_timeslot_free(spot_schedules[spot_id], arrival, departure):
                    assigned_spot = spot_id
                    spot_schedules[spot_id].append((arrival, departure))
                    break

            if assigned_spot != -1:
                initial_soc = round(random.uniform(0.1, 0.5), 2)
                session = EVChargeSession(
                    ev_id=generated_count,
                    spot_id=assigned_spot,
                    arrival_hour=arrival,
                    departure_hour=departure,
                    initial_soc=initial_soc
                )
                self.daily_sessions.append(session)
                generated_count += 1

        print(f"尝试生成 {num_evs_to_generate} 个EV场景，成功创建 {len(self.daily_sessions)} 个。")

        if generated_count < num_evs_to_generate:
            print(f"【警告】: 仅成功生成 {generated_count} / {num_evs_to_generate} 个EV场景。可能是充电桩已满或时段冲突。")
        else:
            print(f"成功生成 {len(self.daily_sessions)} 个EV场景。")

    def get_scenario_for_baseline(self):
        """
        将生成的场景数据转换为baseline优化模型所需的格式。
        修正了BOC（电池荷电状态）的传递逻辑，确保初始SOC被正确填充。
        """
        arrival_times = [[] for _ in range(self.num_spots)]
        departure_times = [[] for _ in range(self.num_spots)]
        present_cars = np.zeros((self.num_spots, 24), dtype=int)

        # 初始化一个25个时间点的BOC数组 (t=0 to t=24)
        boc_initial = np.zeros((self.num_spots, 25))

        # 第一步：标记车辆在场时段，并在到达时刻记录初始SOC
        for session in self.daily_sessions:
            spot, arr, dep = session.spot_id, session.arrival_hour, session.departure_hour

            # 确保不会索引越界
            arr = min(arr, 23)
            dep = min(dep, 24)

            if dep > arr:
                arrival_times[spot].append(arr)
                departure_times[spot].append(dep)
                present_cars[spot, arr:dep] = 1
                boc_initial[spot, arr] = session.initial_soc


        # 第二步：向前填充BOC。如果车辆在某个小时在场，但BOC为0，
        # 就用上一个小时的BOC值来填充，确保SOC数据是连续的。
        for spot in range(self.num_spots):
            for hour in range(1, 25):  # 从 t=1 开始检查
                # 如果当前小时有车在场 (present_cars只到23, 所以用hour-1)
                # 且当前BOC是空的(0)，但前一小时BOC不是空的
                if hour < 24 and present_cars[spot, hour - 1] == 1 and boc_initial[spot, hour] == 0:
                    boc_initial[spot, hour] = boc_initial[spot, hour - 1]

        # 为了让外部代码能像访问chargym环境一样访问参数，创建一个简单的模拟对象
        class MockOriginalEnv:
            def __init__(self, station):
                self.number_of_cars = station.num_spots
                self.EV_Param = {
                    'EV_capacity': station.ev_params.capacity_kwh,
                    'charging_rate': station.ev_params.max_charge_kw,
                    'discharging_rate': station.ev_params.max_discharge_kw,
                    'charging_effic': station.ev_params.charge_efficiency,
                    'discharging_effic': station.ev_params.discharge_efficiency,
                }
                self.Invalues = {
                    'present_cars': present_cars,
                    'BOC': boc_initial,  # 使用修正后的BOC数组
                    'ArrivalT': arrival_times,
                    'DepartureT': departure_times,
                }

        return MockOriginalEnv(self)
