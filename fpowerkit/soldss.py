# 文件: fpowerkit/soldss.py

# (import statements)
import numpy as np
import math
from fpowerkit.grid import Grid
from .solbase import *


# 文件: fpowerkit/soldss.py
# (import statements)

def interrogate_opendss_generators(dss, audit_stage=""):
    """
    【终极Debug函数】审问OpenDSS内部的每一个发电机对象，打印其所有关键参数。
    """
    print("\n" + "#" * 25 + f" INTERROGATING OpenDSS GENERATORS ({audit_stage}) " + "#" * 25)

    if dss.generators.count == 0:
        print("  No Generator objects found in the OpenDSS model.")
        print("#" * 85 + "\n")
        return

    # 定义我们关心的所有关键参数
    properties_to_check = [
        "bus1", "phases", "kv", "kw", "kvar", "pf", "model",
        "vpu", "vmaxpu", "vminpu", "enabled"
    ]

    dss.generators.first()
    while True:
        gen_name = dss.generators.name
        print(f"\n--- Interrogating Generator: {gen_name} ---")

        # 打印关键参数
        for prop in properties_to_check:
            try:
                # 使用dss.text接口来查询属性，这是最可靠的方式
                value = dss.text(f"? Generator.{gen_name}.{prop}")
                print(f"  - {prop:<10}: {value}")
            except Exception as e:
                print(f"  - {prop:<10}: FAILED_TO_QUERY ({e})")

        # 打印实时功率
        try:
            powers = dss.cktelement.powers  # (P, Q) in kW, kVar
            print(f"  - powers (kW, kVar): ({powers[0]:.2f}, {powers[1]:.2f})")
        except Exception as e:
            print(f"  - powers     : FAILED_TO_QUERY ({e})")

        if not dss.generators.next() > 0:
            break

    print("#" * 85 + "\n")
def debug_opendss_internals(dss, sb_kva):
    """
    【终极Debug函数】直接查询OpenDSS内部状态，生成最权威的功率审计报告。
    """
    print("\n" + "#" * 30 + " INTERNAL OpenDSS STATE AUDIT (Post-Solve) " + "#" * 30)

    # 1. 累加所有负荷的实际功率
    total_load_kw = 0
    if dss.loads.count > 0:
        dss.loads.first()
        while True:
            # cktelement.powers 返回 [P1, Q1, P2, Q2, ...]，单位是kW/kVar
            load_powers = dss.cktelement.powers
            total_load_kw += load_powers[0]
            if not dss.loads.next() > 0:
                break

    # 2. 累加所有发电机（不含Vsource）的实际出力
    total_gen_kw = 0
    if dss.generators.count > 0:
        dss.generators.first()
        while True:
            gen_powers = dss.cktelement.powers
            # 电源出力的P值为负，因此取反得到正值
            total_gen_kw += -gen_powers[0]
            if not dss.generators.next() > 0:
                break

    # 3. 获取Vsource（即电网购电）的实际出力
    dss.circuit.set_active_element("Vsource.source")
    vsource_powers = dss.cktelement.powers
    vsource_kw = -vsource_powers[0]

    # 4. 获取总线路损耗
    losses_W, _ = dss.circuit.losses
    losses_kw = losses_W / 1000

    # 5. 打印最终的内部审计报告
    print(f"  --- Power Sources (Injections) ---")
    print(f"  {'Vsource (Grid Inflow)':<30}: {vsource_kw:10.2f} kW")
    print(f"  {'Sum of ALL Generators':<30}: {total_gen_kw:10.2f} kW")
    print(f"  ------------------------------------")
    total_injection_kw = vsource_kw + total_gen_kw
    print(f"  {'TOTAL INJECTION (from DSS)':<30}: {total_injection_kw:10.2f} kW")

    print(f"\n  --- Power Sinks (Ejections) ---")
    print(f"  {'Sum of ALL Loads':<30}: {total_load_kw:10.2f} kW")
    print(f"  {'Total Line Losses':<30}: {losses_kw:10.2f} kW")
    print(f"  ------------------------------------")
    total_ejection_kw = total_load_kw + losses_kw
    print(f"  {'TOTAL EJECTION (from DSS)':<30}: {total_ejection_kw:10.2f} kW")

    balance = total_injection_kw - total_ejection_kw
    print(f"\n  Internal OpenDSS Balance Check: {balance:.2f} kW (should be near 0.0)")
    print("#" * 90 + "\n")


def _convert(il: Island, t: int, source_bus: str):
    """
    【V3.4 逻辑修正+Debug版】
    - 增加了探针C和D，用于追踪Gen和ESS对象的功率值和转换决策。
    """
    try:
        from py_dss_interface import DSS
    except ImportError:
        raise ImportError("py_dss_interface or OpenDSS is not installed...")

    try:
        d = DSS()
    except Exception:
        d = DSS.DSSDLL()

    d.text("clear")
    Ub = il.grid.Ub
    Sb_kVA = il.grid.Sb_kVA
    Zb = il.grid.Zb

    d.text(f"new circuit.my_circuit basekv={Ub} pu=1 MVASC3=5000000 5000000 bus1={source_bus}")
    d.text("New Loadshape.const_shape npts=1 mult=[1.0]")

    for bid, bus in il.BusItems():
        p = bus.Pd(t)
        q = bus.Qd(t)
        if abs(p) > 1e-6 or abs(q) > 1e-6:
            d.text(
                f"New Load.base_{bid.lower()} bus1={bid} phases=3 conn=wye kv={Ub} kW={p * Sb_kVA} kvar={q * Sb_kVA} "
                f"vminpu={bus.MinV} vmaxpu={bus.MaxV} yearly=const_shape")

    for lid, line in il.LineItems():
        if not line.active: continue
        d.text(f"New line.{lid.lower()} bus1={line.fBus} bus2={line.tBus} R1={line.R * Zb} X1={line.X * Zb} units=ohm")

    for pid, pvw in il.PVWItems():
        p = pvw.Pr if pvw.Pr is not None else 0
        q = pvw.Qr if pvw.Qr is not None else 0
        if abs(p) > 1e-6 or abs(q) > 1e-6:
            d.text(
                f"New Generator.{pid.lower()} bus1={pvw.BusID} phases=3 kv={Ub} kw={p * Sb_kVA} kvar={q * Sb_kVA} model=1")

    # 【修正】处理常规发电机 和 SOP虚拟发电机
    for gid, gen in il.GenItems():
        if gen.BusID == source_bus: continue

        p = gen.P(t) if callable(gen.P) else gen.P
        q = gen.Q(t) if callable(gen.Q) else gen.Q
        p = p if p is not None else 0.0
        q = q if q is not None else 0.0



        if abs(p) < 1e-6 and abs(q) < 1e-6: continue

        if p >= 0:
            d.text(
                f"New Generator.{gid.lower()} bus1={gen.BusID} phases=3 kv={Ub} kw={p * Sb_kVA} kvar={q * Sb_kVA} model=1")
        else:
            bus = il.grid.Bus(gen.BusID)
            d.text(
                f"New Load.{gid.lower()} bus1={gen.BusID} phases=3 conn=wye kv={Ub} kW={-p * Sb_kVA} kvar={-q * Sb_kVA} "
                f"vminpu={bus.MinV} vmaxpu={bus.MaxV} yearly=const_shape")


    # 【修正】处理储能系统 (ESS)
    for eid, ess in il.ESSItems():
        p = ess.P if ess.P is not None else 0.0
        q = ess.Q if ess.Q is not None else 0.0

        if abs(p) < 1e-6 and abs(q) < 1e-6: continue

        if p <= 0:
            d.text(
                f"New Generator.{eid.lower()} bus1={ess.BusID} phases=3 kv={Ub} kw={-p * Sb_kVA} kvar={-q * Sb_kVA} model=1")
        else:
            bus = il.grid.Bus(ess.BusID)
            d.text(
                f"New Load.{eid.lower()} bus1={ess.BusID} phases=3 conn=wye kv={Ub} kW={p * Sb_kVA} kvar={q * Sb_kVA} "
                f"vminpu={bus.MinV} vmaxpu={bus.MaxV} yearly=const_shape")

    d.text("set mode=snapshot")
    return d


class OpenDSSSolver(SolverBase):
    def UpdateGrid(self, grid: Grid):
        super().UpdateGrid(grid)
        self.__sbus:'list[str]' = []
        for il in self.Islands:
            b = self.source_buses.intersection(il.Buses)
            assert len(b) == 1, f"Source bus {self.source_buses} not found in an island"
            self.__sbus.append(b.pop())

    def __init__(self, grid:Grid, eps:float = 1e-6, max_iter:int = 1000, *,
            default_saveto:str = DEFAULT_SAVETO, source_bus:'Union[str,Iterable[str]]'):
        if isinstance(source_bus, str): source_bus = [source_bus]
        assert isinstance(source_bus, Iterable), "source_bus must be a string or a list of strings"
        self.source_buses = set(source_bus)
        super().__init__(grid, eps, max_iter, default_saveto = default_saveto)

    def audit_inputs(self, t: int):
        """
        [新增的诊断功能]
        为给定的时间步配置DSS对象（不进行求解），并返回一个详细的字典，
        其中包含OpenDSS视角下的所有负荷和发电机的参数。
        """
        # 假设在审计时只有一个孤岛，这对于baseline运行是典型情况
        if not self.Islands:
            self.UpdateGrid(self.grid)
        il = self.Islands[0]
        il_no = 0
        sbus = self.__sbus[il_no]

        # 调用 _convert 函数来获取已配置好的 DSS 对象
        dss = _convert(il, t, sbus)

        audit_data = {bus.ID: {'loads': [], 'gens': []} for bus in il.grid.Buses}

        # 审计所有负荷 (Loads)
        if dss.loads.count > 0:
            dss.loads.first()
            while True:
                load_name = dss.loads.name
                # 获取挂接母线的名称，并去除可能的节点信息 (如 .1.2.3)
                bus_name_full = dss.cktelement.bus_names[0]
                bus_name = bus_name_full.split('.')[0]
                load_info = {
                    'name': load_name,
                    'kW': dss.loads.kw,
                    'kVar': dss.loads.kvar,
                }
                if bus_name in audit_data:
                    audit_data[bus_name]['loads'].append(load_info)
                if not dss.loads.next() > 0:
                    break

        # 审计所有发电机 (Generators)
        if dss.generators.count > 0:
            dss.generators.first()
            while True:
                gen_name = dss.generators.name
                bus_name_full = dss.cktelement.bus_names[0]
                bus_name = bus_name_full.split('.')[0]
                gen_info = {
                    'name': gen_name,
                    'kW': dss.generators.kw,
                    'kVar': dss.generators.kvar,
                }
                if bus_name in audit_data:
                    audit_data[bus_name]['gens'].append(gen_info)
                if not dss.generators.next() > 0:
                    break

        return audit_data

    def solve_island(self, il_no: int, il: Island, _t: int, /, *, timeout_s: float = 1) -> 'tuple[IslandResult, float]':
        self.dss = _convert(il, _t, self.__sbus[il_no])
        self.dss.text(f"set Voltagebases=[{self.grid.Ub}]")
        self.dss.text("calcv")
        self.dss.text("solve maxcontrol=10000")

        # 1. 回读电压 (逻辑不变)
        if hasattr(self.dss, "circuit"):
            bnames = self.dss.circuit.buses_names
            bvolt = np.array(self.dss.circuit.buses_volts).reshape(-1, 3, 2)
        else:
            bnames = self.dss.circuit_all_bus_names()
            bvolt = np.array(self.dss.circuit_all_bus_volts()).reshape(-1, 3, 2)

        sb_theta = 0
        for i, bn in enumerate(bnames):
            v1 = bvolt[i, 0][0] + 1j * bvolt[i, 0][1]
            v2 = bvolt[i, 1][0] + 1j * bvolt[i, 1][1]
            v = v1 - v2
            b = self.grid.Bus(bn)
            b._v = abs(v) / self.grid.Ub / 1000
            b.theta = math.atan2(v.imag, v.real)
            if bn == self.__sbus[il_no]:
                sb_theta = b.theta
        for i, bn in enumerate(bnames):
            self.grid.Bus(bn).theta -= sb_theta

        Sb_kVA = il.grid.Sb_kVA

        # 2. 回读线路潮流
        for _, line in il.LineItems():
            if not line.active: continue
            self.dss.circuit.set_active_element(f"Line.{line.ID.lower()}")

            # ▼▼▼▼▼ 【核心修复 1：正确累加三相线路潮流】 ▼▼▼▼▼
            powers = self.dss.cktelement.powers  # 返回 [P1, Q1, P2, Q2, P3, Q3]
            # 累加所有相的有功功率和无功功率
            total_p_kw = powers[0] + powers[2] + powers[4]
            total_q_kvar = powers[1] + powers[3] + powers[5]
            line.P = total_p_kw / Sb_kVA
            line.Q = total_q_kvar / Sb_kVA
            # ▲▲▲▲▲ 【修复结束】 ▲▲▲▲▲

        # 3. 回读平衡节点功率
        self.dss.circuit.set_active_element("Vsource.source")
        source_powers_kw_per_phase = self.dss.cktelement.powers

        # ▼▼▼▼▼ 【核心修复 2：正确累加三相购电功率】 ▼▼▼▼▼
        # Vsource出力的P值为负，累加后取反得到正的购电功率
        slack_power_kw = -(
                    source_powers_kw_per_phase[0] + source_powers_kw_per_phase[2] + source_powers_kw_per_phase[4])
        slack_power_pu = slack_power_kw / Sb_kVA
        # ▲▲▲▲▲ 【修复结束】 ▲▲▲▲▲

        try:
            gens_on_slack = self.grid.GensAtBus(self.__sbus[il_no])
            slack_gen_obj = next((g for g in gens_on_slack if hasattr(g, 'is_virtual') and g.is_virtual), None)
            if slack_gen_obj is None:
                slack_gen_obj = next((g for g in gens_on_slack if 'gen_for_slack_bus' in g.ID), None)
            if slack_gen_obj:
                slack_gen_obj._p = slack_power_pu
        except Exception as e:
            print(f"--- OpenDSS Solver 错误: 在回写平衡节点功率时出错: {e}")

        p_loss_kw, q_loss_kvar = self.dss.circuit.losses
        return IslandResult.OK, p_loss_kw