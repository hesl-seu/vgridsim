from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from .island import *
from .grid import *

DEFAULT_SAVETO = "./fpowerkit_logs/"

class GridSolveResult(IntEnum):
    '''Result of grid solving'''
    Failed = 0
    OK = 1
    OKwithoutVICons = 2 # Deprecated
    SubOKwithoutVICons = 3 # Deprecated
    PartialOK = 4


class SolverBase(ABC):
    def __init__(self, grid:Grid, eps:float = 1e-6, max_iter:int = 1000, *, default_saveto:str = DEFAULT_SAVETO, **kwargs):
        self.UpdateGrid(grid)
        self.eps = eps
        self.max_iter = max_iter
        self.saveto = default_saveto
    
    @property
    def grid(self) -> Grid:
        '''Get the grid'''
        return self._g
    @grid.setter
    def grid(self, grid:Grid):
        '''Set the grid'''
        self.UpdateGrid(grid)
        
    def UpdateGrid(self, grid:Grid):
        '''Update the grid'''
        self._g = grid
        self._islands = Island.from_grid(grid)
    
    @property
    def Islands(self) -> 'list[Island]':
        '''Get the islands of the grid'''
        return self._islands
    
    def SetErrorSaveTo(self, path:str = DEFAULT_SAVETO):
        self.saveto = path
        Path(path).mkdir(parents=True, exist_ok = True)

    def _calc_line_params(self):
        for l in self.grid.Lines:
            if not l.active:
                l.I = 0.0
                l.P = 0.0
                l.Q = 0.0
                continue
            vi = self.grid.Bus(l.fBus).V_cpx
            vj = self.grid.Bus(l.tBus).V_cpx
            assert vi is not None and vj is not None
            i = (vi-vj)/l.Z
            l.I = abs(i)
            s = vi*i.conjugate()
            l.P = s.real
            l.Q = s.imag
    
    def solve(self, _t:int, /, **kwargs):
        ok_cnt = 0
        res_val = 0.0
        for i, il in enumerate(self._islands):
            r, v = self.solve_island(i, il, _t, **kwargs)
            il.result = r
            il.result_value = v
            if r == IslandResult.OK:
                ok_cnt += 1
                res_val += v
        self._calc_line_params()
        if ok_cnt == len(self._islands):
            return GridSolveResult.OK, res_val
        elif ok_cnt == 0:
            return GridSolveResult.Failed, -1
        else:
            return GridSolveResult.PartialOK, res_val

    @abstractmethod
    def solve_island(self, i:int, island:Island, _t:int, /, **kwargs) -> 'tuple[IslandResult, float]':
        '''Solve the island'''
        raise NotImplementedError