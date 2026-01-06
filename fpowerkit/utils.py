from typing import Optional, Union
from feasytools import TimeFunc, SegFunc, ConstFunc
from xml.etree.ElementTree import Element


FloatVar = Optional[float]
NFloat = Optional[float]
FloatLike = Union[int, float, TimeFunc, 'list[tuple[int, float]]']
NFloatLike = Union[None, FloatLike]
FloatVals = Union[int, float]


def Float2Func(v: FloatLike) -> TimeFunc:
    if isinstance(v, (float, int)):
        return ConstFunc(v)
    elif isinstance(v, TimeFunc):
        return v
    else:
        return SegFunc(v)

def Func2Elem(f: TimeFunc, tag: str, mul:float = 1, unit:str = "") -> Element:
    if isinstance(f, ConstFunc):
        e = Element(tag,{
            "const": str(f(0)*mul) + unit
        })
    elif isinstance(f, SegFunc):
        e = f.toXMLNode(tag, "item", "time", "value", lambda t, v: f"{v*mul:.4f}{unit}")
    else:
        raise ValueError("Unknown function type")
    return e


def FVstr(s: FloatVar): return "<unsolved>" if s is None else str(s)

def ReadVal(s: str) -> 'tuple[float, str]':
    if s.endswith("pu"):
        return float(s[:-2]), "pu"
    elif s.endswith("kVA"):
        return float(s[:-3]), "kVA"
    elif s.endswith("kvar"):
        return float(s[:-4]), "kvar"
    elif s.endswith("kW"):
        return float(s[:-2]), "kW"
    elif s.endswith("MVA"):
        return float(s[:-3]), "MVA"
    elif s.endswith("Mvar"):
        return float(s[:-4]), "Mvar"
    elif s.endswith("MW"):
        return float(s[:-2]), "MW"
    elif s.endswith("kV"):
        return float(s[:-2]), "kV"
    elif s.endswith("V"):
        return float(s[:-1]), "V"
    elif s.endswith("kA"):
        return float(s[:-2]), "kA"
    elif s.endswith("ohm"):
        return float(s[:-3]), "ohm"
    elif s.endswith("$/puh"):
        return float(s[:-5]), "$/puh"
    elif s.endswith("$/puh2"):
        return float(s[:-6]), "$/puh2"
    elif s.endswith("$"):
        return float(s[:-1]), "$"
    elif s.endswith("$/kWh"):
        return float(s[:-5]), "$/kWh"
    elif s.endswith("$/MWh"):
        return float(s[:-5]), "$/MWh"
    elif s.endswith("$/kWh2"):
        return float(s[:-6]), "$/kWh2"
    elif s.endswith("$/MWh2"):
        return float(s[:-6]), "$/MWh2"
    elif s.endswith("kWh"):
        return float(s[:-3]), "kWh"
    elif s.endswith("MWh"):
        return float(s[:-3]), "MWh"
    elif s.endswith("kWh2"):
        return float(s[:-4]), "kWh2"
    elif s.endswith("MWh2"):
        return float(s[:-4]), "MWh2"
    else:
        return float(s), ""
    
def _valconv(v:FloatVals, u:str, sb_mva, ub_kv) -> FloatVals:
        if u == "pu":
            return v
        elif u == "kVA":
            return v / (sb_mva * 1000)
        elif u == "kvar":
            return v / (sb_mva * 1000)
        elif u == "kW":
            return v / (sb_mva * 1000)
        elif u == "MVA":
            return v / sb_mva
        elif u == "Mvar":
            return v / sb_mva
        elif u == "MW":
            return v / sb_mva
        elif u == "kV":
            return v / ub_kv
        elif u == "V":
            return v / (ub_kv * 1000)
        elif u == "kA":
            return v / (sb_mva / (ub_kv * 3 ** 0.5))
        elif u == "ohm":
            return v * ub_kv ** 2 / sb_mva
        elif u == "$/puh" or u == "$/puh2" or u == "$":
            return v
        elif u == "$/kWh":
            return v * (sb_mva * 1000)
        elif u == "$/MWh":
            return v * sb_mva
        elif u == "$/kWh2":
            return v * (sb_mva * 1000 * sb_mva * 1000)
        elif u == "$/MWh2":
            return v * (sb_mva * sb_mva)
        else:
            return v

def ReadConst(s:str, sb_mva:float, ub_kv:float) -> float:
    v, u = ReadVal(s)
    if u == "": u = "pu"
    return _valconv(v, u, sb_mva, ub_kv)

def ReadNFloatLike(e: Optional[Element], sb_mva:float, ub_kv:float) -> NFloatLike:
    if e is None: return None
    if "const" in e.attrib:
        v, u = ReadVal(e.attrib["const"])
        if u == "": u = e.attrib.get("unit","")
        return _valconv(v, u, sb_mva, ub_kv)
    else:
        repeat = int(e.attrib.get("repeat", "1"))
        period = int(e.attrib.get("period", "0"))
        sf = SegFunc()
        for itm in e:
            time = int(itm.attrib["time"])
            v, u = ReadVal(itm.attrib["value"])
            sf.add(time, _valconv(v, u, sb_mva, ub_kv))
        return sf.repeat(repeat, period)

def ReadFloatLike(e: Optional[Element], sb_mva:float, ub_kv:float) -> FloatLike:
    r = ReadNFloatLike(e, sb_mva, ub_kv)
    assert r is not None
    return r