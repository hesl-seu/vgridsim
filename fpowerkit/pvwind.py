import math
from .utils import *

class PVWind:
    '''
    Model of PV Panel and Wind Turbine
        Such generators are modeled as no operation cost.
        Active power output is a given function of time.
        Reactive power output is calculated from the power factor.
        Not all the power output is used, the maximum power abondon rate is given.
    '''
    def __init__(self, id: str, busid:str, p: FloatLike, pf: float, cc:float, tag: str, Lat: NFloat=None, Lon: NFloat=None):
        '''
        Initialize
            id: PV Panel ID
            busid: ID of the bus where the PV panel is located
            p: Active power output, pu
            pf: Power factor, cos(phi). Reactive output Q = P * sqrt(1 - PF**2)
            cc: curtail cost, $/puh
            tag: "PV" or "Wind", indicating the type of the generator
            Lat: Latitude of the PV Panel
            Lon: Longitude of the PV Panel
        '''
        self._id = id
        self._bus = busid
        self._p = Float2Func(p)
        self._pf = pf
        self._cc = cc
        self._tag = tag
        self._cr:FloatVar = None
        self._pr:FloatVar = None
        self._qr:FloatVar = None
        self.Lat = Lat
        self.Lon = Lon
    
    @property
    def ID(self):
        '''Name of the PV Panel'''
        return self._id
    
    @property
    def BusID(self):
        '''ID of the bus where the PV panel is located'''
        return self._bus
    
    @property
    def P(self):
        '''Active power output, pu'''
        return self._p
    @P.setter
    def P(self, p: FloatLike):
        self._p = Float2Func(p)
    
    @property
    def PF(self):
        '''Power factor, cos(phi). Reactive output Q = P * sqrt(1 - PF**2)'''
        return self._pf
    @PF.setter
    def PF(self, pf: float):
        self._pf = pf
    
    def Q(self, _t):
        '''Reactive power output, pu'''
        return self.P(_t) * math.sqrt(1 - self._pf**2)
    
    def position(self):
        '''Return the position of the PV Panel'''
        return (self.Lat, self.Lon)
    
    def LonLat(self):
        '''Return the position of the PV Panel'''
        return (self.Lon, self.Lat)
    
    def __repr__(self) -> str:
        return (f"PVPanel(id='{self.ID}', busid='{self.BusID}', P={self.P}, PF={self.PF})")

    def __str__(self) -> str:
        return repr(self)
    
    def str_t(self, _t: int) -> str:
        return (f"PVPanel(id='{self.ID}', busid='{self.BusID}', P={self.P(_t)}, PF={self.PF})")
    
    @property
    def Tag(self) -> str:
        '''Return the type of the generator'''
        return self._tag
    @Tag.setter
    def Tag(self, tag: str):
        '''Set the type of the generator'''
        self._tag = tag
        
    @property
    def CR(self):
        '''Curtailment rate'''
        return self._cr
    
    @property
    def CC(self):
        '''Curtailment cost, $/puh'''
        return self._cc
    @CC.setter
    def CC(self, cc: float):
        '''Set the curtailment cost, $/puh'''
        self._cc = cc

    @property
    def Pr(self):
        '''Real power output'''
        return self._pr
    
    @property
    def Qr(self):
        '''Reactive power output'''
        return self._qr
    
    @staticmethod
    def fromXML(node: 'Element', Sb_MVA: float, Ub_kV: float) -> 'PVWind':
        '''
        Load PV Panel from XML node
            node: XML node
        '''
        id = node.attrib["ID"]
        busid = node.attrib["Bus"]
        p = ReadFloatLike(node.find("P"), Sb_MVA, Ub_kV)
        pf = float(node.attrib["pf"])
        cc = ReadConst(node.attrib["cc"], Sb_MVA, Ub_kV) if "cc" in node.attrib else 0
        tag = node.tag
        lat = float(node.attrib["Lat"]) if "Lat" in node.attrib else None
        lon = float(node.attrib["Lon"]) if "Lon" in node.attrib else None
        return PVWind(id, busid, p, pf, cc, tag, lat, lon)

    def toXMLNode(self, Sb_MVA:Optional[float] = None, Sb_kVA:Optional[float] = None) -> 'Element':
        '''Convert to XML node'''
        e = Element(self._tag, {
            "ID": self.ID,
            "Bus": self.BusID,
            "pf": str(self.PF),
        })
        if self.Lat is not None: e.attrib["Lat"] = f"{self.Lat:.6f}"
        if self.Lon is not None: e.attrib["Lon"] = f"{self.Lon:.6f}"
        if Sb_MVA is not None and Sb_kVA is not None:
            raise ValueError("Provide only ONE of Sb_MVA or Sb_kVA.")
        if Sb_MVA is not None:
            mul = Sb_MVA
            unit = ["MW", "$/MWh"]
        elif Sb_kVA is not None:
            mul = Sb_kVA
            unit = ["kW", "$/kWh"]
        else:
            mul = 1
            unit = ["pu", "$/puh"]
        e.append(Func2Elem(self.P, "P", mul, unit[0]))
        e.attrib["cc"] = str(self.CC*mul) + unit[1]
        
        return e
    