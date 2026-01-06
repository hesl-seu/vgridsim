from copy import deepcopy
from .utils import *

NFloatOrFunc = Union[None, float, TimeFunc]

class Generator:
    Pmin: Optional[TimeFunc]  #pu
    Pmax: Optional[TimeFunc]  #pu
    Qmin: Optional[TimeFunc]  #pu
    Qmax: Optional[TimeFunc]  #pu
    CostA: TimeFunc  #$/(pu Power·h)**2
    CostB: TimeFunc  #$/pu Power·h
    CostC: TimeFunc  #$

    def __init__(self, id: str, busid: str, 
            costA: FloatLike, costB: FloatLike, costC: FloatLike,
            pmin_pu: NFloatLike = None, pmax_pu: NFloatLike = None, 
            qmin_pu: NFloatLike = None, qmax_pu: NFloatLike = None,
            Lat: NFloat = None, Lon: NFloat = None,
            P: NFloatLike = None, Q: NFloatLike = None, 
            ):
        '''
        Initialize
            id: Generator ID
            busid: ID of the bus where the generator is located
            costA: Secondary cost, $/(pu Power·h)**2
            costB: Primary cost, $/pu Power·h
            costC: Fixed cost, $
            Lat: Latitude of the generator
            Lon: Longitude of the generator
            Provide either group of parameters:
                pmin_pu: Minimal active power output, pu. None for fixed power
                pmax_pu: Maximal active power output, pu. None for fixed power
            Or:
                P: Active power output, pu. None for unfixed power
            Provide either group of parameters:
                qmin_pu: Minimal reactive power output, pu. None for fixed power
                qmax_pu: Maximal reactive power output, pu. None for fixed power
            Or:
                Q: Reactive power output, pu. None for unfixed power
        '''
        self._id = id
        self._bus = busid
        self.Lat = Lat
        self.Lon = Lon
        if P is not None:
            self._p = Float2Func(P)
            self._fixed_p = True
            self.Pmin = None
            self.Pmax = None
        elif pmin_pu is not None and pmax_pu is not None:
            self._p = None
            self._fixed_p = False
            self.Pmin = Float2Func(pmin_pu)
            self.Pmax = Float2Func(pmax_pu)
        else:
            raise ValueError("Either P or (pmin, pmax) should be provided.")
        if Q is not None:
            self._q = Float2Func(Q)
            self._fixed_q = True
            self.Qmin = None
            self.Qmax = None
        elif qmin_pu is not None  and qmax_pu is not None:
            self._q = None
            self._fixed_q = False
            self.Qmin = Float2Func(qmin_pu)
            self.Qmax = Float2Func(qmax_pu)
        else:
            raise ValueError("Either Q or (qmin, qmax) should be provided.")
        self.CostA = Float2Func(costA)
        self.CostB = Float2Func(costB)
        self.CostC = Float2Func(costC)
        self.CostShadow = None
    
    @property
    def P(self) -> NFloatOrFunc:
        '''Active power output, pu'''
        return self._p
    
    @property
    def FixedP(self) -> bool:
        '''Whether the active power output is fixed'''
        return self._fixed_p
    
    def fixP(self, p: float):
        '''Fix the active power output'''
        self._p = p
        self._fixed_p = True
    
    def unfixP(self):
        '''Unfix the active power output'''
        self._p = None
        self._fixed_p = False
    
    @property
    def Q(self) -> NFloatOrFunc:
        '''Reactive power output, pu'''
        return self._q
    
    @property
    def FixedQ(self) -> bool:
        '''Whether the reactive power output is fixed'''
        return self._fixed_q
    
    def fixQ(self, q: float):
        '''Fix the reactive power output'''
        self._q = q
        self._fixed_q = True
    
    def unfixQ(self):
        '''Unfix the reactive power output'''
        self._q = None
        self._fixed_q = False
    
    @property
    def position(self) -> 'tuple[NFloat, NFloat]':
        '''Position of the generator, (Lat, Lon)'''
        return (self.Lat, self.Lon)
    
    @property
    def LonLat(self) -> 'tuple[NFloat, NFloat]':
        return self.Lon, self.Lat
    
    @property
    def ID(self) -> str:
        '''Name of the generator'''
        return self._id
    
    @property
    def BusID(self) -> str:
        '''ID of the bus where the generator is located'''
        return self._bus
    
    def __repr__(self) -> str:
        return (f"Generator(id='{self.ID}', busid='{self.BusID}', P={self.P}, Q={self.Q}, " +
                f"pmin_pu={self.Pmin}, pmax_pu={self.Pmax}, qmin_pu={self.Qmin}, qmax_pu={self.Qmax}, " + 
                f"costA={self.CostA}, costB={self.CostB}, costC={self.CostC}, Lat={self.Lat}, Lon={self.Lon})")

    def __str__(self) -> str:
        return repr(self)

    def str_t(self, _t: int, /) -> str:
        p = self.P(_t) if isinstance(self.P, TimeFunc) else FVstr(self.P)
        pmin = self.Pmin(_t) if isinstance(self.Pmin, TimeFunc) else None
        pmax = self.Pmax(_t) if isinstance(self.Pmax, TimeFunc) else None
        q = self.Q(_t) if isinstance(self.Q, TimeFunc) else FVstr(self.Q)
        qmin = self.Qmin(_t) if isinstance(self.Qmin, TimeFunc) else None
        qmax = self.Qmax(_t) if isinstance(self.Qmax, TimeFunc) else None
        return (f"Generator(id='{self.ID}', busid='{self.BusID}', P={p}, Q={q}, " + 
                f"pmin_pu={pmin}, pmax_pu={pmax}, qmin_pu={qmin}, qmax_pu={qmax}, " + 
                f"costA={self.CostA(_t)}, costB={self.CostB(_t)}, costC={self.CostC(_t)}, Lat={self.Lat}, Lon={self.Lon})")

    def Cost(self, _t: int, /, secondary: bool = True) -> FloatVar:
        '''
        Get the cost of the generator at time _t, $/h
            _t: time
            secondary: whether to use the secondary cost model, False for the primary cost model
        '''
        if self.P is None: return None
        p = self.P(_t) if isinstance(self.P, TimeFunc) else self.P
        ret = self.CostB(_t) * p + self.CostC(_t)
        if secondary: ret += self.CostA(_t) * p ** 2
        return ret

    def CostPerPUPower(self, _t: int, /, secondary: bool = True) -> FloatVar:
        '''
        Get the cost of the generator per pu power at time _t, $/pu Power·h
            _t: time
            secondary: whether to use the secondary cost model, False for the primary cost model
        '''
        if self.P is None: return None
        p = self.P(_t) if isinstance(self.P, TimeFunc) else self.P
        if p == 0: return None
        ret = self.CostB(_t) * p + self.CostC(_t)
        if secondary: ret += self.CostA(_t) * p ** 2
        return ret / p

    @staticmethod
    def fromXML(node: 'Element', Sb_MVA: float, Ub_kV: float):
        id = node.attrib["ID"]
        busid = node.attrib["Bus"]
        pmin = ReadNFloatLike(node.find("Pmin"), Sb_MVA, Ub_kV)
        pmax = ReadNFloatLike(node.find("Pmax"), Sb_MVA, Ub_kV)
        qmin = ReadNFloatLike(node.find("Qmin"), Sb_MVA, Ub_kV)
        qmax = ReadNFloatLike(node.find("Qmax"), Sb_MVA, Ub_kV)
        ca = ReadFloatLike(node.find("CostA"), Sb_MVA, Ub_kV)
        cb = ReadFloatLike(node.find("CostB"), Sb_MVA, Ub_kV)
        cc = ReadFloatLike(node.find("CostC"), Sb_MVA, Ub_kV)
        p = ReadNFloatLike(node.find("P"), Sb_MVA, Ub_kV)
        q = ReadNFloatLike(node.find("Q"), Sb_MVA, Ub_kV)
        lat = node.attrib.get("Lat", None)
        if lat is not None: lat = float(lat)
        lon = node.attrib.get("Lon", None)
        if lon is not None: lon = float(lon)
        return Generator(id, busid, ca, cb, cc, pmin, pmax, qmin, qmax, lat, lon, p, q)

    def toXMLNode(self, Sb_MVA:Optional[float] = None, Sb_kVA:Optional[float] = None) -> 'Element':
        e = Element("gen", {
            "ID": self.ID,
            "Bus": self.BusID,
        })
        if self.Lat is not None: e.attrib["Lat"] = f"{self.Lat:.6f}"
        if self.Lon is not None: e.attrib["Lon"] = f"{self.Lon:.6f}"
        if Sb_MVA is not None and Sb_kVA is not None:
            raise ValueError("Provide only ONE of Sb_MVA or Sb_kVA.")
        if Sb_MVA is not None:
            mul = Sb_MVA
            unit = ["MW", "Mvar", "$/MWh2", "$/MWh", "$"]
        elif Sb_kVA is not None:
            mul = Sb_kVA
            unit = ["kW", "kvar", "$/kWh2", "$/kWh", "$"]
        else:
            mul = 1
            unit = ["pu", "pu", "$/puh2", "$/puh", "$"]
        if self.Pmin: e.append(Func2Elem(self.Pmin, "Pmin", mul, unit[0]))
        if self.Pmax: e.append(Func2Elem(self.Pmax, "Pmax", mul, unit[0]))
        if self.Qmin: e.append(Func2Elem(self.Qmin, "Qmin", mul, unit[1]))
        if self.Qmax: e.append(Func2Elem(self.Qmax, "Qmax", mul, unit[1]))
        e.append(Func2Elem(self.CostA, "CostA", 1 / (mul*mul), unit[2]))
        e.append(Func2Elem(self.CostB, "CostB", 1 / mul, unit[3]))
        e.append(Func2Elem(self.CostC, "CostC", 1, unit[4]))
        return e
    
    
class GeneratorModel:
    Pmin: TimeFunc  #pu
    Pmax: TimeFunc  #pu
    Qmin: TimeFunc  #pu
    Qmax: TimeFunc  #pu
    CostA: TimeFunc  #$/(pu Power·h)**2
    CostB: TimeFunc  #$/pu Power·h
    CostC: TimeFunc  #$
    def __init__(self, pmin_pu: FloatLike, pmax_pu: FloatLike, qmin_pu: FloatLike, qmax_pu: FloatLike,
                 costA: FloatLike, costB: FloatLike, costC: FloatLike):
        '''
        Initialize
            pmin_pu: Minimal active power output, pu
            pmax_pu: Maximal active power output, pu
            qmin_pu: Minimal reactive power output, pu
            qmax_pu: Maximal reactive power output, pu
            costA: Secondary cost, $/(pu Power·h)**2
            costB: Primary cost, $/pu Power·h
            costC: Fixed cost, $
        '''
        self.Pmin = Float2Func(pmin_pu)
        self.Pmax = Float2Func(pmax_pu)
        self.Qmin = Float2Func(qmin_pu)
        self.Qmax = Float2Func(qmax_pu)
        self.CostA = Float2Func(costA)
        self.CostB = Float2Func(costB)
        self.CostC = Float2Func(costC)
    
    def toGenerator(self, id:str, busid:str):
        return Generator(id, busid, deepcopy(self.CostA), deepcopy(self.CostB), 
            deepcopy(self.CostC), deepcopy(self.Pmin), deepcopy(self.Pmax), 
            deepcopy(self.Qmin), deepcopy(self.Qmax))
    
    @staticmethod
    def fromXML(node: 'Element', Sb_MVA: float, Ub_kV: float):
        pmin = ReadFloatLike(node.find("Pmin"), Sb_MVA, Ub_kV)
        pmax = ReadFloatLike(node.find("Pmax"), Sb_MVA, Ub_kV)
        qmin = ReadFloatLike(node.find("Qmin"), Sb_MVA, Ub_kV)
        qmax = ReadFloatLike(node.find("Qmax"), Sb_MVA, Ub_kV)
        ca = ReadFloatLike(node.find("CostA"), Sb_MVA, Ub_kV)
        cb = ReadFloatLike(node.find("CostB"), Sb_MVA, Ub_kV)
        cc = ReadFloatLike(node.find("CostC"), Sb_MVA, Ub_kV)
        return GeneratorModel(pmin, pmax, qmin, qmax, ca, cb, cc)

    def toXMLNode(self) -> 'Element':
        e = Element("genmodel")
        e.append(Func2Elem(self.Pmin, "Pmin"))
        e.append(Func2Elem(self.Pmax, "Pmax"))
        e.append(Func2Elem(self.Qmin, "Qmin"))
        e.append(Func2Elem(self.Qmax, "Qmax"))
        e.append(Func2Elem(self.CostA, "CostA"))
        e.append(Func2Elem(self.CostB, "CostB"))
        e.append(Func2Elem(self.CostC, "CostC"))
        return e