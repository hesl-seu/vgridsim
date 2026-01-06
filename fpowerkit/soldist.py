from itertools import chain
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
import math
import warnings
from .solbase import *
from .island import *

VF = Union['Var', float]  # type: ignore

@dataclass
class LoadReduceModule:
    '''Load Reduce module'''
    Bus: BusID
    Limit: TimeFunc
    Reduction: FloatVar = None

class DistFlowSolver(SolverBase):
    '''DistFlow solver'''
    def __init__(self, grid:Grid, eps:float = 1e-6, max_iter:int = 1000, *, 
            default_saveto:str = DEFAULT_SAVETO, mlrp: float = 0.5, secondary_cost: bool = True, bigM: float = 1e6):
        '''
        Initialize
            grid: Grid object
            default_saveto: Default path to save the results
            mlrp: Maximum proportion of load reduction
            secondary_cost: Whether to include the secondary cost. 
                If False, the cost is calculated as CostB * Pg + CostC;
                If True, the cost is calculated as CostA * Pg^2 + CostB * Pg + CostC.
        '''
        super().__init__(grid, eps, max_iter, default_saveto = default_saveto)
        self._decb: dict[BusID, LoadReduceModule] = {}
        self.C = bigM
        self._mlrp = mlrp
        self._oflines: set[LineID] = set()
        self._ofbuses: set[BusID] = set()
        self._sec_cost = secondary_cost     
    
    @property
    def OverflowLines(self):
        return self._oflines
    
    @property
    def OverflowBuses(self):
        return self._ofbuses
    
    def UpdateGrid(self, grid:Grid, cut_overflow_lines: bool = False):
        if cut_overflow_lines:
            for lID in self._oflines:
                self.grid.Line(lID).active = False
        super().UpdateGrid(grid)
        self.__il_relax = [False] * len(self._islands)
    
    @property
    def MLRP(self):
        '''Get the maximum load reduction proportion'''
        return self._mlrp
    @MLRP.setter
    def MLRP(self, v: float):
        '''Set the maximum load reduction proportion'''
        if v < 0 or v > 1:
            raise ValueError("Invalid maximum load reduction proportion")
        self._mlrp = v
    
    def AddReduce(self, bus: BusID, limit: TimeFunc, reduction: Optional[FloatVar] = None):
        '''Add a load reduction module'''
        self._decb[bus] = LoadReduceModule(bus, limit, reduction)
    
    def RemoveReduce(self, bus: BusID):
        '''Remove a load reduction module'''
        if bus in self._decb:
            del self._decb[bus]
        
    def GetReduce(self, bus: BusID) -> LoadReduceModule:
        '''Get the load reduction module'''
        return self._decb[bus]
    
    @property
    def DecBuses(self):
        return self._decb
    
    def solve(self, _t: int, /, *, timeout_s: float = 1) -> 'tuple[GridSolveResult, float]':
        '''Get the best result at time _t, return a tuple: (result status, optimal objective value)'''
        allOK = True
        allFail = True
        val = 0
        for i, il in enumerate(self._islands):
            relax = self.__il_relax[i]
            r, v = self.solve_island(i, il, _t, timeout_s, relax, relax)
            self._islands[i].result = r
            self._islands[i].result_value = v
            if r == IslandResult.Failed:
                self.__il_relax[i] = True
                r, v = self.solve_island(i, il, _t, timeout_s, True, True)
                self._islands[i].result = r
                self._islands[i].result_value = v
            if r != IslandResult.Failed:
                allFail = False
                val += v
            if r != IslandResult.OK: allOK = False
            if r == IslandResult.OverFlow: self.__il_relax[i] = False
        if allOK:
            return GridSolveResult.OK, val
        elif allFail:
            print(f"Failed to solve at time {_t}")
            if self.saveto != "":
                p = Path(self.saveto)
                p.mkdir(parents=True, exist_ok=True)
                self.grid.savePQofBus(str(p/f"{_t}_load.csv"), _t)
            return GridSolveResult.Failed, val
        else:
            return GridSolveResult.PartialOK, val
    
    def __chkv(self, d: Union[None, np.ndarray]) -> float:
        assert d is not None, "Value is None"
        assert isinstance(d, (np.ndarray, np.floating, float)), f"Value is {type(d)}"
        ret = d.item()
        assert isinstance(ret, float), f"Value is not a float: {type(ret)}"
        assert ret != float('inf'), "Value is inf"
        assert ret != float('-inf'), "Value is -inf"
        return ret

    def solve_island(self, il_idx:int, island: Island, _t: int, timeout_s: float, relaxV: bool, relaxI: bool, clear_results: bool = False) -> 'tuple[IslandResult, float]':
        
        ''' ---------Variables----------
        pg0[k]: Generator active power
        qg0[k]: Generator reactive power
        pvwp[k]: PVWind active power
        --> pg[j]: Active power of all generators at the bus
        --> qg[j]: Reactive power of all generators at the bus
        v[j]: Bus voltage ** 2
        l[i,j]: Line current ** 2
        P[i,j]: Line active power
        Q[i,j]: Line reactive power
        '''
        try:
            import cvxpy as cp
            from cvxpy import Variable as Var
        except ImportError:
            raise ImportError("cvxpy or ecos is not installed. Please install them using 'pip install cvxpy ecos'")
        warnings.filterwarnings("ignore", category=FutureWarning, module="cvxpy")
        cons: 'list[cp.Constraint]' = []
        # Create GEN vars
        pg0: 'dict[str, VF]' = {}
        qg0: 'dict[str, VF]' = {}
        for gID, g in island.GenItems():
            if g.FixedP:
                assert g.P is not None
                pg0[gID] = g.P(_t) if isinstance(g.P, TimeFunc) else g.P
            elif g.Pmin is not None and g.Pmax is not None:
                v = Var(name=f"pg_{gID}")
                cons.append(v >= g.Pmin(_t))
                cons.append(v <= g.Pmax(_t))
                pg0[gID] = v
            else:
                raise ValueError(f"Generator {gID} provides neither P or (pmin, pmax)")
            if g.FixedQ:
                assert g.Q is not None
                qg0[gID] = g.Q(_t) if isinstance(g.Q, TimeFunc) else g.Q
            elif g.Qmin is not None and g.Qmax is not None:
                v = Var(name=f"qg_{gID}")
                cons.append(v >= g.Qmin(_t))
                cons.append(v <= g.Qmax(_t))
                qg0[gID] = v
            else:
                raise ValueError(f"Generator {g.ID} provides neither Q or (qmin, qmax)")
        
        pvwp: dict[str, Var] = {
            pID: Var(name=f"pvw_{pID}", nonneg=True) 
            for pID, p in island.PVWItems()
        }
        pvwq: dict[str, Var] = {
            pID: Var(name=f"pvwq_{pID}")
            for pID in island.PVWs
        }
        
        # Add PVW constraints
        for pID, p in island.PVWItems():
            cons.append(pvwp[pID] <= p.P(_t))
            cons.append(pvwp[pID] * math.sqrt(1 - p.PF**2) == pvwq[pID])

        # Bind GEN vars to Bus
        pg: 'dict[str, list[VF]]' = {b: [] for b in island.Buses}
        qg: 'dict[str, list[VF]]' = {b: [] for b in island.Buses}
        pd: 'dict[str, float]' = {bID: b.Pd(_t) for bID, b in island.BusItems()}
        qd: 'dict[str, float]' = {bID: b.Qd(_t) for bID, b in island.BusItems()}
        for gID, g in island.GenItems():
            pg[g.BusID].append(pg0[gID])
            qg[g.BusID].append(qg0[gID])
        for pID, p in island.PVWItems():
            pg[p.BusID].append(pvwp[pID])
            qg[p.BusID].append(pvwq[pID])
        for eID, e in island.ESSItems():
            p, q = e.GetLoad(_t, island.grid.ChargePrice(_t), island.grid.DischargePrice(_t))
            e.P = p
            if p > 0:
                pd[e.BusID] += p
                qd[e.BusID] += q
            elif p < 0:
                pg[e.BusID].append(-p)
                qg[e.BusID].append(-q)
        
        # Create BUS vars
        v = {bID: Var(name=f"v_{bID}") for bID in island.Buses}
        dvmin: dict[str, Var] = {}
        dvmax: dict[str, Var] = {}
        for bid, b in island.BusItems():
            if b.FixedV:
                assert b.V is not None, f"Bus {bid} has fixed voltage but not set"
                cons.append(v[bid] == b.V ** 2)
            elif relaxV:
                dvmin[bid] = Var(name=f"dvmin_{bid}", nonneg=True)
                dvmax[bid] = Var(name=f"dvmax_{bid}", nonneg=True)
                cons.append(v[bid] >= b.MinV ** 2 - dvmin[bid])
                cons.append(v[bid] <= b.MaxV ** 2 + dvmax[bid])
            else:
                cons.append(v[bid] >= b.MinV ** 2)
                cons.append(v[bid] <= b.MaxV ** 2)
        
        # Create Line vars
        dlmax: dict[str, Var] = {}
        l = {lID: Var(name=f"l_{lID}", nonneg=True) for lID in island.Lines}
        
        for lID, ln in island.LineItems():
            if ln.max_I == math.inf:
                continue
            if relaxI:
                dlmax[lID] = Var(name=f"dlmax_{lID}", nonneg=True)
                cons.append(l[lID] <= (ln.max_I/island.grid.Ib) ** 2 + dlmax[lID])
            else:
                cons.append(l[lID] <= (ln.max_I/island.grid.Ib) ** 2)
        
        P = {lID: Var(name=f"P_{lID}") for lID in island.Lines}
        Q = {lID: Var(name=f"Q_{lID}") for lID in island.Lines}

        Pdec: dict[BusID, Var] = {
            bus: Var(name=f"Pdec_{bus}", nonneg=True)
            for bus, lim in self._decb.items()
        }
        for bus, lim in self._decb.items():
            cons.append(Pdec[bus] <= lim.Limit(_t) * self._mlrp)
        
        # ----------Constraints-----------
        for j, bus in island.BusItems():
            flow_in = island.grid.LinesOfTBus(j)
            flow_out = island.grid.LinesOfFBus(j)
            dec = Pdec[j] if j in Pdec else 0
            
            # P constraint
            inflow = cp.sum([P[ln.ID] - ln.R * l[ln.ID] for ln in flow_in if ln.ID in island.Lines])
            outflow = cp.sum([P[ln.ID] for ln in flow_out if ln.ID in island.Lines])
            cons.append(
                inflow + cp.sum(pg[j]) == outflow + pd[j] - dec # type: ignore
            )

            # flow_in and flow_out are Python generators, which cannot be reused, thus needed to be re-assigned
            flow_in = island.grid.LinesOfTBus(j)
            flow_out = island.grid.LinesOfFBus(j)
            # Q constraint
            q_inflow = cp.sum([Q[ln.ID] - ln.X * l[ln.ID] for ln in flow_in if ln.ID in island.Lines])
            q_outflow = cp.sum([Q[ln.ID] for ln in flow_out if ln.ID in island.Lines])
            cons.append(
                q_inflow + cp.sum(qg[j]) == q_outflow + qd[j] # type: ignore
            )

        for lid, line in island.LineItems():
            i, j = line.pair
            lid = line.ID
            cons.append(
                v[j] == v[i] - 2 * (line.R * P[lid] + line.X * Q[lid]) + (line.R ** 2 + line.X ** 2) * l[lid]
            )
            cons.append(
                cp.quad_over_lin(cp.hstack([P[lid], Q[lid]]), v[i]) <= l[lid]
            )
        
        for pID, p in island.PVWItems():
            cons.append(pvwp[pID] + cp.sqrt(1 - p.PF**2) == pvwq[pID])

        # Objective components
        decs = self.C * (cp.sum(list(Pdec.values())) + 
                        (cp.sum(list(dvmin.values())) if relaxV else 0) + 
                        (cp.sum(list(dvmax.values())) if relaxV else 0) + 
                        (cp.sum(list(dlmax.values())) if relaxI else 0))
        
        crpe = cp.sum([p.CC*(p.P(_t)-pvwp[pID]) for pID, p in island.PVWItems()])
        
        if self._sec_cost:
            goal = cp.sum([g.CostA(_t) * cp.square(pg0[gID]) + g.CostB(_t) * pg0[gID] + g.CostC(_t) 
                         for gID, g in island.GenItems()])
        else:
            goal = cp.sum([g.CostB(_t) * pg0[gID] + g.CostC(_t) 
                         for gID, g in island.GenItems()])

        problem = cp.Problem(cp.Minimize(decs + goal + crpe), cons)

        try:
            problem.solve(solver=cp.ECOS, verbose=False, max_iters=self.max_iter)
        except cp.SolverError as e:
            if "installed" in e.args[0]:
                raise e
            return IslandResult.Failed, -1

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            for _, bus in island.BusItems():
                if not bus.FixedV: bus._v = 0
                bus.ShadowPrice = 0
            for _, p in island.PVWItems():
                p._pr = p._qr = p._cr = 0
            for _, line in island.LineItems():
                line.I = line.P = line.Q = 0
            for j, gen in island.GenItems():
                if not gen.FixedP: gen._p = 0
                if not gen.FixedQ: gen._q = 0
            for _, e in island.ESSItems():
                e.P = 0
            return IslandResult.Failed, -1

        for j, bus in island.BusItems():
            bus._v = self.__chkv(v[j].value) ** 0.5

        for lid, line in island.LineItems():
            line.I = self.__chkv(l[lid].value) ** 0.5
            line.P = self.__chkv(P[lid].value)
            line.Q = self.__chkv(Q[lid].value)

        for j, gen in island.GenItems():
            p = pg0[j]
            if isinstance(p, Var) and p.value is not None: gen._p = self.__chkv(p.value)
            q = qg0[j]
            if isinstance(q, Var): gen._q = self.__chkv(q.value)
        
        for pID, p in island.PVWItems():
            p._pr = self.__chkv(pvwp[pID].value)
            p._qr = self.__chkv(pvwq[pID].value)
            pgen = p.P(_t)
            p._cr = 1 - p._pr / pgen if pgen > 0 else 0
        
        overflow = False
        self._ofbuses.clear()
        if relaxV:
            for bID, bv in chain(dvmax.items(), dvmin.items()):
                if self.__chkv(bv.value) > 1e-8:
                    overflow = True
                    self._ofbuses.add(bID)
        
        self._oflines.clear()
        if relaxI:
            for lID, lv in dlmax.items():
                if self.__chkv(lv.value) > 1e-8:
                    overflow = True
                    self._oflines.add(lID)
        
        for bus, lim in self._decb.items():
            lim.Reduction = self.__chkv(Pdec[bus].value)
            if lim.Reduction < 1e-8: lim.Reduction = 0

        if not isinstance(goal, (int, float)):
            goal = self.__chkv(goal.value)
        return IslandResult.OverFlow if overflow else IslandResult.OK, goal