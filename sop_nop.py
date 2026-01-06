# file: sop_nop.py
"""
该模块定义了软开关(SOP)和常开点(NOP)的数据类。

这两个类主要用作数据容器，存储从配置文件（如 asexcel 文件）中读取的
SOP 和 NOP 设备的物理参数。仿真和优化模块将使用这些类的实例来构建电网模型。
"""

from typing import Optional
from xml.etree.ElementTree import Element
from fpowerkit.utils import NFloat


class SOP:
    """
    软开关 (Soft Open Point, SOP) 类。

    SOP 是一种先进的电力电子设备，用于在配电网中灵活地控制潮流。
    它通常被看作是一个理想的可控源，可以连接两个不同的馈线或母线，
    并精确地注入或吸收指定大小的有功和无功功率，从而实现负载均衡、电压调节等功能。
    """

    def __init__(self, id: str, bus1: str, bus2: str, p_max_pu: float, q_max_pu: float, loss_coeff: float = 0.05,
                 active: bool = True):
        """
        初始化SOP对象。

        参数:
            id (str): SOP的唯一标识符。
            bus1 (str): 连接的第一个母线的ID。
            bus2 (str): 连接的第二个母线的ID。
            p_max_pu (float): 能够传输或吸收的最大有功功率 (标幺值)。
            q_max_pu (float): 能够传输或吸收的最大无功功率 (标幺值)。
            loss_coeff (float): 功率损耗系数，用于估算SOP自身的运行损耗。
            active (bool): 标记该SOP在当前仿真中是否可用。
        """
        self._id = id
        self._bus1 = bus1
        self._bus2 = bus2
        self._p_max = p_max_pu
        self._q_max = q_max_pu
        self._loss_coeff = loss_coeff
        self.active = active

        # 以下属性用于在求解器运行后存储结果
        self.P1 = None  # 在bus1侧的有功功率 (pu)
        self.Q1 = None  # 在bus1侧的无功功率 (pu)
        self.P2 = None  # 在bus2侧的有功功率 (pu)
        self.Q2 = None  # 在bus2侧的无功功率 (pu)

    @property
    def ID(self) -> str:
        """获取SOP的唯一标识符ID。"""
        return self._id

    @property
    def Bus1(self) -> str:
        """获取SOP连接的第一个母线的ID。"""
        return self._bus1

    @property
    def Bus2(self) -> str:
        """获取SOP连接的第二个母线的ID。"""
        return self._bus2

    @property
    def PMax(self) -> float:
        """获取SOP的最大有功功率传输能力 (pu)。"""
        return self._p_max

    @property
    def QMax(self) -> float:
        """获取SOP的最大无功功率传输能力 (pu)。"""
        return self._q_max

    @property
    def LossCoeff(self) -> float:
        """获取SOP的损耗系数。"""
        return self._loss_coeff

    def __repr__(self) -> str:
        """返回对象的字符串表示形式，方便调试时打印查看。"""
        return f"SOP(id='{self.ID}', bus1='{self.Bus1}', bus2='{self.Bus2}', p_max_pu={self.PMax}, q_max_pu={self.QMax}, active={self.active})"


class NOP:
    """
    常开点 (Normally Open Point, NOP) 类。

    NOP 是配电网中的一个联络开关，它在正常运行时处于断开状态。
    在需要进行网络重构以实现故障恢复、负载均衡或网损最优化时，
    可以通过闭合一个或多个NOP来改变网络的拓扑结构。
    """

    def __init__(self, id: str, bus1: str, bus2: str, r_pu: float, x_pu: float, max_I_kA: float = float('inf'),
                 active: bool = False):
        """
        初始化NOP对象。

        参数:
            id (str): NOP的唯一标识符。
            bus1 (str): 连接的第一个母线的ID。
            bus2 (str): 连接的第二个母线的ID。
            r_pu (float): 当NOP闭合时，等效线路的电阻 (标幺值)。
            x_pu (float): 当NOP闭合时，等效线路的电抗 (标幺值)。
            max_I_kA (float): NOP闭合时允许通过的最大电流 (kA)。
            active (bool): NOP的初始状态，False表示断开 (默认)，True表示闭合。
        """
        self._id = id
        self._bus1 = bus1
        self._bus2 = bus2
        self._r = r_pu
        self._x = x_pu
        self._max_I = max_I_kA
        self.active = active  # 该状态将在优化过程中由决策变量决定

        # 以下属性用于在求解器运行后存储结果
        self.P = None  # 闭合时流过的有功功率 (pu)
        self.Q = None  # 闭合时流过的无功功率 (pu)
        self.I = None  # 闭合时流过的电流 (pu)

    @property
    def ID(self) -> str:
        """获取NOP的唯一标识符ID。"""
        return self._id

    @property
    def Bus1(self) -> str:
        """获取NOP连接的第一个母线的ID。"""
        return self._bus1

    @property
    def Bus2(self) -> str:
        """获取NOP连接的第二个母线的ID。"""
        return self._bus2

    @property
    def R(self) -> float:
        """获取NOP闭合时的等效电阻 (pu)。"""
        return self._r

    @property
    def X(self) -> float:
        """获取NOP闭合时的等效电抗 (pu)。"""
        return self._x

    @property
    def MaxI(self) -> float:
        """获取NOP的最大电流容量 (kA)。"""
        return self._max_I

    def __repr__(self) -> str:
        """返回对象的字符串表示形式，方便调试时打印查看。"""
        return f"NOP(id='{self.ID}', bus1='{self.Bus1}', bus2='{self.Bus2}', r_pu={self.R}, x_pu={self.X}, max_I_kA={self.MaxI}, active={self.active})"