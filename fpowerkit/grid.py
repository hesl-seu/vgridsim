from collections import deque
from enum import IntEnum
import gzip
from xml.etree.ElementTree import ElementTree
import zipfile
from typing import Iterable

import numpy as np
from .bus import *
from .line import *
from .gen import *
from .pvwind import *
from .ess import *

BusID = str
LineID = str

class PQSaveMode(IntEnum):
    PU = 0
    KVA = 1
    MVA = 2

class USaveMode(IntEnum):
    PU = 0
    KV = 1

class ZSaveMode(IntEnum):
    PU = 0
    OHM = 1


class Grid:
    Sb: float  #MVA
    Ub: float  #kV

    @property
    def Sb_MVA(self) -> float:
        return self.Sb

    @property
    def Sb_kVA(self) -> float:
        return self.Sb * 1000

    @property
    def Zb(self) -> float:
        '''Zb, unit = Ohm'''
        return self.Ub ** 2 / self.Sb

    @property
    def Ib(self) -> float:
        '''Ib, unit = kA'''
        return self.Sb / (self.Ub * (3 ** 0.5))

    def __init__(self, Sb_MVA: float, Ub_kV: float, buses: 'Iterable[Bus]', lines: 'Iterable[Line]',
            gens: 'Iterable[Generator]', pvws: 'Optional[Iterable[PVWind]]' = None, esss: 'Optional[Iterable[ESS]]' = None, 
            cprice: FloatLike = 0, dprice: FloatLike = 0, holdShadowPrice: bool = False):
        '''
        Initialize
            Sb_MVA: base power, MVA
            Ub_kV: base voltage, kV
            buses: list of buses
            lines: list of lines
            gens: list of generators
            pvws: list of PVWinds
            esss: list of ESSs
            cprice: charge price of the whole grid, $/puh
            dprice: discharge price of the whole grid, $/puh
            holdShadowPrice: whether to hold the shadow price of the buses of the last time when failed to solve this time
        '''
        self.Sb = Sb_MVA
        self.Ub = Ub_kV
        self._busl = list(buses)
        self._buses = {bus.ID: bus for bus in self._busl}
        self._lines = {line.ID: line for line in lines}        
        self._gens = {gen.ID: gen for gen in gens}
        if pvws is None: pvws = []
        self._pvws = {pvw.ID: pvw for pvw in pvws}
        if esss is None: esss = []
        self._esss = {ess.ID: ess for ess in esss}
        self._cp = Float2Func(cprice)
        self._dp = Float2Func(dprice)
        self._holdShadowPrice = holdShadowPrice   

        self._bnames = list(self._buses.keys())
        self._ladjfb: 'dict[str, list[Line]]' = {bus.ID: [] for bus in self._buses.values()}
        self._ladjtb: 'dict[str, list[Line]]' = {bus.ID: [] for bus in self._buses.values()}
        self._gatb: 'dict[str, list[Generator]]' = {bus.ID: [] for bus in self._buses.values()}
        self._patb: 'dict[str, list[PVWind]]' = {bus.ID: [] for bus in self._buses.values()}
        self._eatb: 'dict[str, list[ESS]]' = {bus.ID: [] for bus in self._buses.values()}
        for line in self._lines.values():
            if not line.fBus in self._bnames: raise ValueError(f"Bus {line.fBus} undefined")
            if not line.tBus in self._bnames: raise ValueError(f"Bus {line.tBus} undefined")
            self._ladjfb[line.fBus].append(line)
            self._ladjtb[line.tBus].append(line)
        for gen in self._gens.values():
            assert gen.BusID in self._bnames, f"Bus {gen.BusID} undefined"
            self._gatb[gen.BusID].append(gen)
        for pvw in self._pvws.values():
            assert pvw.BusID in self._bnames, f"Bus {pvw.BusID} undefined"
            self._patb[pvw.BusID].append(pvw)
        for ess in self._esss.values():
            assert ess.BusID in self._bnames, f"Bus {ess.BusID} undefined"
            self._eatb[ess.BusID].append(ess)

    @property
    def ChargePrice(self):
        return self._cp

    @property
    def DischargePrice(self):
        return self._dp 
    
    def AddGen(self, g: Generator):
        '''Add a generator'''
        self._gens[g.ID] = g
        assert g.BusID in self._bnames, f"Bus {g.BusID} undefined"
        self._gatb[g.BusID].append(g)
    
    def DelGen(self, id: str):
        '''Delete a generator'''
        g = self._gens.pop(id)
        self._gatb[g.BusID].remove(g)
    
    def AddBus(self, b: Bus):
        '''Add a bus'''
        self._buses[b.ID] = b
        self._bnames.append(b.ID)
        self._ladjfb[b.ID] = []
        self._ladjtb[b.ID] = []
        self._gatb[b.ID] = []
        self._patb[b.ID] = []
    
    def DelBus(self, id: str):
        '''Delete a bus and all related lines, generators and PVWinds'''
        b = self._buses.pop(id)
        self._bnames.remove(id)
        for l in self._ladjfb[b.ID]: self._lines.pop(l.ID)
        for l in self._ladjtb[b.ID]: self._lines.pop(l.ID)
        for g in self._gatb[b.ID]: self._gens.pop(g.ID)
        for p in self._patb[b.ID]: self._pvws.pop(p.ID)
        self._ladjfb.pop(b.ID)
        self._ladjtb.pop(b.ID)
        self._gatb.pop(b.ID)
        self._patb.pop(b.ID)
    
    def AddLine(self, l: Line):
        '''Add a line'''
        self._lines[l.ID] = l
        assert l.fBus in self._bnames, f"Bus {l.fBus} undefined"
        assert l.tBus in self._bnames, f"Bus {l.tBus} undefined"
        self._ladjfb[l.fBus].append(l)
        self._ladjtb[l.tBus].append(l)

    def DelLine(self, id: str):
        '''Delete a line'''
        l = self._lines.pop(id)
        self._ladjfb[l.fBus].remove(l)
        self._ladjtb[l.tBus].remove(l)
    
    def AddPVWind(self, p: PVWind):
        '''Add a PVWind'''
        self._pvws[p.ID] = p
        assert p.BusID in self._bnames, f"Bus {p.BusID} undefined"
        self._patb[p.BusID].append(p)
    
    def DelPVWind(self, id: str):
        '''Delete a PVWind'''
        p = self._pvws.pop(id)
        self._patb[p.BusID].remove(p)
    
    def AddESS(self, e: ESS):
        '''Add an ESS'''
        self._esss[e.ID] = e
    
    def ApplyAllESS(self, dur: int):
        '''Apply all ESSs'''
        for e in self._esss.values():
            e.Apply(dur)

    @property
    def BusNames(self) -> 'list[str]':
        return self._bnames

    def Bus(self, id: str) -> 'Bus':
        return self._buses[id]
    
    def ChangePVWindID(self, old_id:str, new_id:str):
        if old_id == new_id: return
        if new_id in self._pvws.keys():
            raise ValueError(f"Duplicated PVWind name: {new_id}")
        p = self._pvws.pop(old_id)
        p._id = new_id
        self._pvws[new_id] = p
    
    def ChangePVWindBus(self, pvw_id:str, new_bus_id:str):
        if new_bus_id not in self._bnames:
            raise ValueError(f"Invalid bus name: {new_bus_id}")
        p = self._pvws.get(pvw_id)
        if not p: raise ValueError(f"Invalid PVWind name: {pvw_id}")
        if p.BusID == new_bus_id: return
        self._patb[p.BusID].remove(p)
        p._bus = new_bus_id
        self._patb[p.BusID].append(p)
    
    def ChangeGenID(self, old_id:str, new_id:str):
        if old_id == new_id: return
        if new_id in self._gens.keys():
            raise ValueError(f"Duplicated generator name: {new_id}")
        g = self._gens.pop(old_id)
        g._id = new_id
        self._gens[new_id] = g
    
    def ChangeGenBus(self, gen_id:str, new_bus_id:str):
        if new_bus_id not in self._bnames:
            raise ValueError(f"Invalid bus name: {new_bus_id}")
        g = self._gens.get(gen_id)
        if not g: raise ValueError(f"Invalid generator name: {gen_id}")
        if g.BusID == new_bus_id: return
        self._gatb[g.BusID].remove(g)
        g._bus = new_bus_id
        self._gatb[g.BusID].append(g)
                                    
    def ChangeLineFromBus(self, line_id:str, new_bus_id:str):
        if new_bus_id not in self._bnames:
            raise ValueError(f"Invalid bus name: {new_bus_id}")
        l = self._lines.get(line_id)
        if not l: raise ValueError(f"Invalid line name: {line_id}")
        if l.fBus == new_bus_id: return
        self._ladjfb[l.fBus].remove(l)
        l._fBus = new_bus_id
        self._ladjfb[l.fBus].append(l)
    
    def ChangeLineToBus(self, line_id:str, new_bus_id:str):
        if new_bus_id not in self._bnames:
            raise ValueError(f"Invalid bus name: {new_bus_id}")
        l = self._lines.get(line_id)
        if not l: raise ValueError(f"Invalid line name: {line_id}")
        if l.tBus == new_bus_id: return
        self._ladjtb[l.tBus].remove(l)
        l._tBus = new_bus_id
        self._ladjtb[l.tBus].append(l)
    
    def ChangeLineID(self, old_id: str, new_id: str):
        if old_id == new_id: return
        if new_id in self._lines.keys():
            raise ValueError(f"Duplicated line name: {new_id}")
        l = self._lines.pop(old_id)
        l._id = new_id
        self._lines[new_id] = l
    
    def ChangeBusID(self, old_id: str, new_id: str):
        if old_id == new_id: return
        if new_id in self._bnames:
            raise ValueError(f"Duplicated bus name: {new_id}")
        b = self._buses.pop(old_id)
        b._id = new_id
        self._buses[new_id] = b
        afb = self._ladjfb.pop(old_id)
        for l in afb: l._fBus = new_id
        self._ladjfb[new_id] = afb
        atb = self._ladjtb.pop(old_id)
        for l in atb: l._tBus = new_id
        self._ladjtb[new_id] = atb
        gb = self._gatb.pop(old_id)
        for g in gb: g._bus = new_id
        self._gatb[new_id] = gb
        self._bnames.remove(old_id)
        self._bnames.append(new_id)

    def ChangeESSID(self, old_id:str, new_id:str):
        if old_id == new_id: return
        if new_id in self._esss.keys():
            raise ValueError(f"Duplicated ESS name: {new_id}")
        e = self._esss.pop(old_id)
        e._id = new_id
        self._esss[new_id] = e
    
    def ChangeESSBus(self, ess_id:str, new_bus_id:str):
        if new_bus_id not in self._bnames:
            raise ValueError(f"Invalid bus name: {new_bus_id}")
        e = self._esss.get(ess_id)
        if not e: raise ValueError(f"Invalid ESS name: {ess_id}")
        if e.BusID == new_bus_id: return
        self._eatb[e.BusID].remove(e)
        e._bus = new_bus_id
        self._eatb[e.BusID].append(e)
    
    @property
    def Buses(self):
        return self._buses.values()

    def LinesOfFBus(self, busid: str, only_active:bool = True) -> 'Iterable[Line]':
        if only_active:
            return (l for l in self._ladjfb[busid] if l.active)
        else:
            return self._ladjfb[busid]

    def LinesOfTBus(self, busid: str, only_active:bool = True) -> 'Iterable[Line]':
        if only_active:
            return (l for l in self._ladjtb[busid] if l.active)
        else:
            return self._ladjtb[busid]

    def Line(self, id: str) -> 'Line':
        return self._lines[id]

    @property
    def Lines(self):
        return self._lines.values()
    
    @property
    def ActiveLines(self):
        return (l for l in self._lines.values() if l.active)

    def Gen(self, id: str) -> 'Generator':
        return self._gens[id]

    @property
    def GenNames(self) -> 'list[str]':
        return list(self._gens.keys())

    @property
    def Gens(self):
        return self._gens.values()

    def GensAtBus(self, busid: str) -> 'list[Generator]':
        return self._gatb[busid]

    @property
    def PVWinds(self):
        return self._pvws.values()
    
    def PVWind(self, id: str) -> 'PVWind':
        return self._pvws[id]
    
    @property
    def PVWindNames(self) -> 'list[str]':
        return list(self._pvws.keys())
    
    @property
    def ESSs(self):
        return self._esss.values()
    
    def ESS(self, id: str) -> 'ESS':
        return self._esss[id]
    
    def ESSsAtBus(self, busid: str) -> 'list[ESS]':
        return self._eatb[busid]
    
    def __repr__(self):
        b = '\n  '.join(map(str, self._buses.values()))
        l = '\n  '.join(map(str, self._lines.values()))
        g = '\n  '.join(map(str, self._gens.values()))
        return f"Ub={self.Ub}kV, Sb={self.Sb_MVA}MVA\nBuses:\n  {b}\nLines:\n  {l}\nGenerators:\n  {g}"

    def __str__(self):
        return repr(self)
    
    def str_t(self, _t: int):
        b = '\n  '.join(v.str_t(_t) for v in self._buses.values())
        l = '\n  '.join(v.str_t(_t) for v in self._lines.values())
        g = '\n  '.join(v.str_t(_t) for v in self._gens.values())
        return f"Ub={self.Ub}kV, Sb={self.Sb_MVA}MVA\nAt time {_t}:\nBuses:\n  {b}\nLines:\n  {l}\nGenerators:\n  {g}"

    @staticmethod
    def fromFile(file_name:str, holdShadowPrice: bool = False):
        return Grid.fromFileXML(file_name, holdShadowPrice)
        
    @staticmethod
    def fromFileXML(xml_name: str, holdShadowPrice: bool = False):
        if xml_name.lower().endswith(".xml.gz"):
            fh = gzip.open(xml_name)
            rt = ElementTree(file=fh).getroot()
            fh.close()
        elif xml_name.lower().endswith(".xml"):
            rt = ElementTree(file=xml_name).getroot()
        else:
            raise ValueError("Unsupported file type")
        if rt is None:
            raise ValueError(f"Invalid XML file: {xml_name}")
        # Read base values
        Sb, unit = ReadVal(rt.attrib["Sb"])
        if unit == "MVA": pass
        elif unit == "kVA": Sb /= 1000
        else: raise ValueError(f"Invalid base value unit: {unit}")
        Ub, unit = ReadVal(rt.attrib["Ub"])
        if unit == "kV": pass
        else: raise ValueError(f"Invalid base value unit: {unit}")
        #Read grid model
        bm = rt.attrib.get("model","")
        if bm != "": # No base model
            try:
                grpt = int(rt.attrib.get("grid-repeat","1"))
                lrpt = int(rt.attrib.get("load-repeat","1"))
                fixed = rt.attrib.get("fixed-load","") == "true"
            except:
                raise ValueError("Invalid grid-repeat or load-repeat or fixed-load")
            from .cases import _get_buses, _get_lines, _get_gens, _DEFAULT_LOAD_SCALE, _DEFAULT_GENERATOR, _DEFAULT_GEN_POS
            if bm.lower() == "ieee33":
                from .cases import _IEEE33_LOAD, _IEEE33_LINE
                buses:list[Bus] = _get_buses(Sb * 1000, grpt, _IEEE33_LOAD, not fixed, _DEFAULT_LOAD_SCALE, lrpt, 86400)
                lines:list[Line] = _get_lines(Ub * Ub / Sb, _IEEE33_LINE, grpt)
                gens:list[Generator] = _get_gens(_DEFAULT_GENERATOR, _DEFAULT_GEN_POS, grpt)
            elif bm.lower() == "ieee69":
                from .cases import _IEEE69_LOAD, _IEEE69_LINE
                buses:list[Bus] = _get_buses(Sb * 1000, grpt, _IEEE69_LOAD, not fixed, _DEFAULT_LOAD_SCALE, lrpt, 86400)
                lines:list[Line] = _get_lines(Ub * Ub / Sb, _IEEE69_LINE, grpt)
                gens:list[Generator] = _get_gens(_DEFAULT_GENERATOR, _DEFAULT_GEN_POS, grpt)
            else:
                raise ValueError(f"Undefined base model: {bm}")
        else:
            buses:list[Bus] = []
            lines:list[Line] = []
            gens:list[Generator] = []
        pvws:list[PVWind] = []
        esss:list[ESS] = []
        #Read buses, lines and generators
        cp = 0
        dp = 0

        gen_models:dict[str, GeneratorModel] = {}
        for e in rt:
            if e.tag == "bus":
                buses.append(Bus.fromXML(e, Sb, Ub))
            elif e.tag == "line":
                lines.append(Line.fromXML(e, Ub * Ub / Sb))
            elif e.tag == "gen" or e.tag == "generator":
                bm = e.attrib.get("model","")
                if bm == "": # No base model
                    gens.append(Generator.fromXML(e, Sb, Ub))
                else:
                    if bm not in gen_models: raise ValueError(f"Undefined generator model: {bm}")
                    gens.append(gen_models[bm].toGenerator(e.attrib["ID"], e.attrib["Bus"]))
            elif e.tag == "genmodel" or e.tag == "generatormodel":
                gen_models[e.attrib["name"]] = GeneratorModel.fromXML(e, Sb, Ub)
            elif e.tag == "pv" or e.tag == "wind":
                pvws.append(PVWind.fromXML(e, Sb, Ub))
            elif e.tag == "ess":
                esss.append(ESS.fromXML(e, Sb, Ub))
            elif e.tag == "cprice":
                cp = ReadFloatLike(e, Sb, Ub)
            elif e.tag == "dprice":
                dp = ReadFloatLike(e, Sb, Ub)
            else:
                raise ValueError(f"Unknown XML node: {e.tag}")
        return Grid(Sb, Ub, buses, lines, gens, pvws, esss, cp, dp, holdShadowPrice)

    def YMat(self) -> 'np.ndarray':
        '''Return the admittance matrix of the grid'''
        n = len(self._buses)
        Y = np.zeros((n, n), dtype=complex)
        for l in self.ActiveLines:
            i = self._bnames.index(l.fBus)
            j = self._bnames.index(l.tBus)
            y = 1 / l.Z
            Y[i, j] -= y
            Y[j, i] -= y
            Y[i, i] += y
            Y[j, j] += y
        return Y
    
    def toXMLNode(self,
            PQmode: PQSaveMode = PQSaveMode.MVA, 
            Umode: USaveMode = USaveMode.PU, 
            Zmode: ZSaveMode = ZSaveMode.OHM
        )->'Element':
        e = Element('grid', {
            "Sb": f"{self.Sb_MVA}MVA",
            "Ub": f"{self.Ub}kV",
        })
        for b in self.Buses:
            e.append(b.toXMLNode(
                Ub_kV = self.Ub if Umode == USaveMode.KV else None,
                Sb_MVA = self.Sb_MVA if PQmode == PQSaveMode.MVA else None,
                Sb_kVA = self.Sb_kVA if PQmode == PQSaveMode.KVA else None
            ))
        for l in self.Lines:
            e.append(l.toXMLNode(
                Zb_Ohm = self.Zb if Zmode == ZSaveMode.OHM else None
            ))
        for g in self.Gens:
            e.append(g.toXMLNode(
                Sb_MVA = self.Sb_MVA if PQmode == PQSaveMode.MVA else None,
                Sb_kVA = self.Sb_kVA if PQmode == PQSaveMode.KVA else None
            ))
        for p in self.PVWinds:
            e.append(p.toXMLNode(
                Sb_MVA = self.Sb_MVA if PQmode == PQSaveMode.MVA else None,
                Sb_kVA = self.Sb_kVA if PQmode == PQSaveMode.KVA else None
            ))
        for es in self.ESSs:
            e.append(es.toXMLNode(
                Sb_MVA = self.Sb_MVA if PQmode == PQSaveMode.MVA else None,
                Sb_kVA = self.Sb_kVA if PQmode == PQSaveMode.KVA else None
            ))
        return e

    def saveFileXML(self, path:str,
            PQmode: PQSaveMode = PQSaveMode.MVA, 
            Umode: USaveMode = USaveMode.PU, 
            Zmode: ZSaveMode = ZSaveMode.OHM
        ):
        e = self.toXMLNode(PQmode, Umode, Zmode)
        et = ElementTree(element=e)
        if path.lower().endswith(".xml"):
            et.write(path, encoding="utf-8")
        elif path.lower().endswith(".xml.gz"):
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
                with z.open("data.xml", "w") as f:
                    et.write(f, encoding="utf-8")
        else:
            raise ValueError("Unsupported file type. Only .xml and .xml.gz are supported")

    def savePQofBus(self, file_name: str, t:int):
        with open(file_name, "w") as fp:
            fp.write("Bus,Pd,Qd\n")
            for bus in self.Buses:
                fp.write(f"{bus.ID},{bus.Pd(t)},{bus.Qd(t)}\n")
    
    def loadPQofBus(self, file_name: str):
        with open(file_name) as fp:
            for ln in fp.readlines()[1:]:
                bn, p, q = ln.strip().split(',')
                self.Bus(bn).Pd = ConstFunc(float(p) / self.Sb)
                self.Bus(bn).Qd = ConstFunc(float(q) / self.Sb)
