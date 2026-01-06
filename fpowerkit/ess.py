from enum import IntEnum, Enum
import math
from feasytools import RangeList
from .utils import *

class ESSPolicy(Enum):
    Manual = "Manual"   # Charging/discharging manually
    Time = "Time"       # Charging/discharging at given period
    Price = "Price"     # Charging at low price, discharging at high price

class ESSManualState(IntEnum):
    Idle = 0
    Charge = 1
    Discharge = 2

class ESS:
    '''Model of an Energy Storage System'''
    def __init__(self, id: str, busid: str, cap_puh: float, ec:float, ed:float, pc_max:float, pd_max:float, 
        pf:float, policy:ESSPolicy, ctime:Optional[RangeList]=None, dtime:Optional[RangeList]=None,
        cprice:NFloat=None, dprice:NFloat=None, Lat: NFloat=None, Lon: NFloat=None, init_elec_puh: float=0.0):
        '''
        Initialize
            id: ESS ID
            busid: ID of the bus where the ESS is located
            cap_puh: Capacity of the ESS, puh
            ec: Charging efficiency
            ed: Discharging efficiency
            pc_max: Maximum charging power, pu
            pd_max: Maximum discharging power, pu
            pf: Power factor, cos(phi). Reactive output Q = P * sqrt(1 - PF**2)
            policy: ESSPolicy
            c_cond: Charging condition
            d_cond: Discharging condition
            Lat: Latitude of the ESS
            Lon: Longitude of the ESS
        '''
        self._id = id
        self._bus = busid
        self._cap = cap_puh
        self._ec = ec
        self._ed = ed
        self._pc_max = pc_max
        self._pd_max = pd_max
        self._pf = pf
        self.Lat = Lat
        self.Lon = Lon
        self._elec = init_elec_puh
        self._policy = policy
        self._ctime = ctime
        self._dtime = dtime
        self._cprice = cprice
        self._dprice = dprice
        self._manstate = ESSManualState.Idle
        self.P:FloatVar = None

    @property
    def Q(self) -> float:
        '''Reactive power output'''
        return self.__q(self.P) if self.P is not None else 0.0

    @property
    def SOC(self) -> float:
        '''State of charge'''
        return self._elec / self._cap
    
    def Apply(self, dur_s:int):
        if self.P is not None:
            dur = dur_s / 3600
            if self.P < 0:
                return self.Discharge(-self.P, dur)
            else:
                return self.Charge(self.P, dur)
        return 0
    
    def __q(self, P: float) -> float:
        '''Calculate reactive power'''
        return P * math.sqrt(1 - self._pf**2)
    
    def GetLoad(self, t: int, cprice:float, dprice:float) -> 'tuple[float, float]':
        '''
        Get the power of the ESS. Return p+jq.
        p > 0 means the ESS is charging, and thus the ESS is treated as a load.
        p < 0 means the ESS is discharging, and thus the ESS is treated as a generator.
        '''
        if self._policy == ESSPolicy.Manual:
            if self._manstate == ESSManualState.Charge and self._elec < self._cap:
                return self._pc_max, self.__q(self._pc_max)
            elif self._manstate == ESSManualState.Discharge and self._elec > 0:
                return -self._pd_max, -self.__q(self._pd_max)
        elif self._policy == ESSPolicy.Time:
            if self._ctime is not None and t in self._ctime and self._elec < self._cap:
                return self._pc_max, self.__q(self._pc_max)
            elif self._dtime is not None and t in self._dtime and self._elec > 0:
                return -self._pd_max, -self.__q(self._pd_max)
        elif self._policy == ESSPolicy.Price:
            if self._cprice is not None and self._dprice is not None:
                if cprice < self._cprice and self._elec < self._cap:
                    return self._pc_max, self.__q(self._pc_max)
                elif dprice > self._dprice and self._elec > 0:
                    return -self._pd_max, -self.__q(self._pd_max)
        else:
            raise ValueError("Unknown ESS policy")
        return 0, 0
    
    def Charge(self, P: float, dt: float):
        '''Charge the ESS with power P for time dt'''
        old = self._elec
        self._elec = min(self._cap, self._elec + P * self._ec * dt)
        return self._elec - old
    
    def Discharge(self, P: float, dt: float):
        '''Discharge the ESS with power P for time dt'''
        old = self._elec
        self._elec = max(0, self._elec - P / self._ed * dt)
        return self._elec - old
    
    @property
    def ID(self):
        '''Name of the ESS'''
        return self._id
    
    @property
    def BusID(self):
        '''ID of the bus where the ESS is located'''
        return self._bus
    
    @property
    def MaxPc(self):
        '''Maximum charging power, pu'''
        return self._pc_max
    @MaxPc.setter
    def MaxPc(self, pc: float):
        self._pc_max = pc

    @property
    def MaxPd(self):
        '''Maximum discharging power, pu'''
        return self._pd_max
    @MaxPd.setter
    def MaxPd(self, pd: float):
        self._pd_max = pd
    
    @property
    def Cap(self):
        '''Capacity of the ESS, puh'''
        return self._cap
    @Cap.setter
    def Cap(self, cap: float):
        self._cap = cap
    
    @property
    def EC(self):
        '''Charging efficiency'''
        return self._ec
    @EC.setter
    def EC(self, ec: float):
        assert 0 <= ec <= 1, "Charging efficiency must be in [0, 1]"
        self._ec = ec
    
    @property
    def ED(self):
        '''Discharging efficiency'''
        return self._ed
    @ED.setter
    def ED(self, ed: float):
        assert 0 <= ed <= 1, "Discharging efficiency must be in [0, 1]"
        self._ed = ed
    
    @property
    def PF(self):
        '''Power factor, cos(phi). Reactive output Q = P * sqrt(1 - PF**2)'''
        return self._pf
    @PF.setter
    def PF(self, pf: float):
        assert -1 <= pf <= 1, "Power factor must be in [-1, 1]"
        self._pf = pf

    def position(self):
        '''Return the position of the ESS'''
        return (self.Lat, self.Lon)

    def LonLat(self):
        '''Return the position of the ESS'''
        return (self.Lon, self.Lat)
    
    def __repr__(self) -> str:
        return (f"ESS(id='{self.ID}', busid='{self.BusID}', "+
            f"pc_max={self.MaxPc:.6f}, pd_max={self.MaxPd:.6f}, pf={self.PF:.6f}, "+
            f"ec={self.EC:.6f}, ed={self.ED:.6f}, Lat={self.Lat:.6f}, Lon={self.Lon:.6f})")

    def __str__(self) -> str:
        return repr(self)
    
    def str_t(self, _t: int) -> str:
        return repr(self)
    
    @staticmethod
    def fromXML(node: 'Element', Sb_MVA: float, Ub_kV: float) -> 'ESS':
        '''
        Load ESS from XML node
            node: XML node
        '''
        id = node.attrib["ID"]
        busid = node.attrib["Bus"]
        cap = ReadConst(node.attrib["cap"], Sb_MVA, Ub_kV)
        ec = float(node.attrib["ec"])
        ed = float(node.attrib["ed"])
        pc_max = ReadConst(node.attrib["pc_max"], Sb_MVA, Ub_kV)
        pd_max = ReadConst(node.attrib["pd_max"], Sb_MVA, Ub_kV)
        pf = float(node.attrib["pf"])
        policy = ESSPolicy(node.attrib["policy"]) if "policy" in node.attrib else ESSPolicy.Manual
        ctime_node = node.find("ctime")
        if ctime_node is not None:
            ctime = RangeList(ctime_node)
        else:
            ctime = None
        dtime_node = node.find("dtime")
        if dtime_node is not None:
            dtime = RangeList(dtime_node)
        else:
            dtime = None
        cprice = ReadConst(node.attrib["cprice"], Sb_MVA, Ub_kV) if "cprice" in node.attrib else None
        dprice = ReadConst(node.attrib["dprice"], Sb_MVA, Ub_kV) if "dprice" in node.attrib else None
        lat = float(node.attrib["Lat"]) if "Lat" in node.attrib else None
        lon = float(node.attrib["Lon"]) if "Lon" in node.attrib else None
        init = ReadConst(node.attrib["init_elec"], Sb_MVA, Ub_kV) if "init_elec" in node.attrib else 0.0
        return ESS(id, busid, cap, ec, ed, pc_max, pd_max, pf, policy, 
            ctime, dtime, cprice, dprice, lat, lon, init)

    def toXMLNode(self, Sb_MVA:Optional[float] = None, Sb_kVA:Optional[float] = None) -> 'Element':
        '''Convert to XML node'''
        e = Element("ess", {
            "ID": self.ID,
            "Bus": self.BusID,
            "ec": str(self.EC),
            "ed": str(self.ED),
            "pf": str(self.PF),
            "policy": self._policy.value
        })
        if self.Lat is not None: e.attrib["Lat"] = f"{self.Lat:.6f}"
        if self.Lon is not None: e.attrib["Lon"] = f"{self.Lon:.6f}"
        if self._ctime is not None:
            e.append(self._ctime.toXMLNode("ctime"))
        if self._dtime is not None:
            e.append(self._dtime.toXMLNode("dtime"))
        if Sb_MVA is not None and Sb_kVA is not None:
            raise ValueError("Provide only ONE of Sb_MVA or Sb_kVA.")
        if Sb_MVA is not None:
            mul = Sb_MVA
            unit = ["MWh", "MW", "$/MWh"]
        elif Sb_kVA is not None:
            mul = Sb_kVA
            unit = ["kWh", "kW", "$/kWh"]
        else:
            mul = 1
            unit = ["puh", "pu", "$/puh"]
        e.attrib["cap"] = str(self._cap*mul) + unit[0]
        e.attrib["pc_max"] = str(self._pc_max*mul) + unit[1]
        e.attrib["pd_max"] = str(self._pd_max*mul) + unit[1]
        e.attrib["init_elec"] = str(self._elec*mul) + unit[0]
        if self._cprice: e.attrib["cprice"] = str(self._cprice*mul) + unit[2]
        if self._dprice: e.attrib["dprice"] = str(self._dprice*mul) + unit[2]
        return e