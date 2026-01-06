import numpy as np
import math
from dataclasses import dataclass
from .solbase import *

class BusType(IntEnum):
    PQ = 0
    PV = 1
    Slack = 2

@dataclass
class _NRProblem:
    eqs: 'list[int]'
    n_P: int
    Ps: 'list[float]'
    Qs: 'list[float]'
    bus_dict: 'dict[str, int]'
    G: 'np.ndarray'
    B: 'np.ndarray'
    V: 'list[float]'
    theta: 'list[float]'

def _presolve(il:Island, _t:int):
    '''Check the bus type'''
    busType: 'dict[BusID, BusType]' = {}
    slack_cnt = 0
    eq_P: 'list[int]' = []
    eq_Q: 'list[int]' = []
    V: 'list[float]' = []
    theta: 'list[float]' = []
    Ps: 'list[float]' = []
    Qs: 'list[float]' = []
    bus_dict, Y = il.YMat()
    for bid, i in bus_dict.items():
        bus = il.grid.Bus(bid)
        V.append(bus.V if bus.V is not None else 1.0)
        theta.append(bus.theta if bus.theta is not None else 0.0)
        fixp = fixq = True
        p = -bus.Pd(_t)
        q = -bus.Qd(_t)
        for g in il.grid.GensAtBus(bus.ID):
            if not g.FixedP: fixp = False
            else:
                assert g.P is not None
                if isinstance(g.P, TimeFunc): p += g.P(_t)
                else: p += g.P
            if not g.FixedQ: fixq = False
            else:
                assert g.Q is not None
                if isinstance(g.Q, TimeFunc): q += g.Q(_t)
                else: q += g.Q
        if bus.FixedV:
            if fixp and fixq:
                raise ValueError(f"Bus {bus.ID}: Invalid bus type PQV")
            elif fixp:
                busType[bus.ID] = BusType.PV
                eq_P.append(i)
            elif fixq:
                raise ValueError(f"Bus {bus.ID}: Invalid bus type VQ")
            else:
                busType[bus.ID] = BusType.Slack
                slack_cnt += 1
                if slack_cnt > 1:
                    raise ValueError('Only one slack bus is allowed')
        else:
            if fixp and fixq:
                busType[bus.ID] = BusType.PQ
                bus.V = 1.0
                eq_P.append(i)
                eq_Q.append(i)
            elif fixp:
                raise ValueError(f"Bus {bus.ID}: Invalid bus type: Pθ")
            elif fixq:
                raise ValueError(f"Bus {bus.ID}: Invalid bus type: Qθ")
            else:
                raise ValueError(f"Bus {bus.ID}: Invalid bus type: θ")
        Ps.append(p)
        Qs.append(q)
    if slack_cnt == 0:
        raise ValueError('No slack bus is found')
    return _NRProblem(eq_P + eq_Q, len(eq_P), Ps, Qs, bus_dict, Y.real, Y.imag, V, theta)

def _solve(prb: _NRProblem, max_iter:int = 100, eps:float = 1e-6):
    '''Solve the power flow problem'''
    G = prb.G
    B = prb.B
    V = prb.V
    T = prb.theta
    eqs = prb.eqs
    Ps = prb.Ps
    Qs = prb.Qs
    n_P = prb.n_P
    n = len(prb.eqs)
    m = len(prb.bus_dict)

    def P(i:int):
        return V[i]*sum(V[j]*(
            G[i,j]*math.cos(T[i]-T[j]) + B[i,j]*math.sin(T[i]-T[j])
        ) for j in range(m))
    
    def Q(i:int):
        return V[i]*sum(V[j]*(
            G[i,j]*math.sin(T[i]-T[j]) - B[i,j]*math.cos(T[i]-T[j])
        ) for j in range(m))
    
    def H(i:int, j:int) -> float:
        if i == j: return Q(i)+V[i]**2*B[i,i]
        return -V[i]*V[j]*(G[i,j]*math.sin(T[i]-T[j]) - B[i,j]*math.cos(T[i]-T[j]))
    
    def N(i:int, j:int) -> float:
        if i == j: return -P(i)-V[i]**2*G[i,i]
        return -V[i]*V[j]*(G[i,j]*math.cos(T[i]-T[j]) + B[i,j]*math.sin(T[i]-T[j]))
    
    def M(i:int, j:int) -> float:
        if i == j: return -P(i)+V[i]**2*G[i,i]
        return V[i]*V[j]*(G[i,j]*math.cos(T[i]-T[j]) + B[i,j]*math.sin(T[i]-T[j]))
    
    def L(i:int, j:int) -> float:
        if i == j: return -Q(i)+V[i]**2*B[i,i]
        return -V[i]*V[j]*(G[i,j]*math.sin(T[i]-T[j]) - B[i,j]*math.cos(T[i]-T[j]))
    
    cnt = 0
    while cnt < max_iter:
        y = np.zeros(n, dtype=np.float64)
        for i, b in enumerate(eqs):
            y[i] = Ps[b] - P(b) if i < n_P else Qs[b] - Q(b)
        if np.abs(y).max() < eps:
            break
        cnt += 1
        J = np.zeros((n, n), dtype=np.float64)
        for i, b0 in enumerate(eqs):
            for j, b1 in enumerate(eqs):
                if i < n_P and j < n_P: #dp/dθ
                    J[i,j] = H(b0, b1)
                elif i < n_P and j >= n_P: #V*dp/dV
                    J[i,j] = N(b0, b1)
                elif i >= n_P and j < n_P: #dp/dV
                    J[i,j] = M(b0, b1)
                elif i >= n_P and j >= n_P: #V*dq/dV
                    J[i,j] = L(b0, b1)
        x = np.linalg.solve(J, -y)
        for i, (x0, b) in enumerate(zip(x, eqs)):
            if i < n_P: # Δθ
                T[b] += x0
            else: # ΔU/U
                V[b] += V[b] * x0
    if cnt >= max_iter:
        raise ValueError("Bad solution")
    return cnt, V, T, prb.bus_dict

class NewtonSolver(SolverBase):
    def __init__(self, grid:Grid, eps:float = 1e-6, default_saveto:str = DEFAULT_SAVETO, max_iter:int = 100):
        super().__init__(grid, eps, max_iter, default_saveto = default_saveto)

    def solve_island(self, i:int, island:Island, _t:int, **kwargs) -> 'tuple[IslandResult, float]':
        try:
            prb = _presolve(island, _t)
            cnt, V, theta, bus_dict = _solve(prb, self.max_iter, self.eps)
        except ValueError as e:
            return IslandResult.Failed, 0.0
        
        for b in bus_dict.keys():
            bus = island.grid.Bus(b)
            bus.V = V[bus_dict[b]]
            bus.theta = theta[bus_dict[b]]
        return IslandResult.OK, 0.0