from collections import deque
from .grid import *

class IslandResult(IntEnum):
    Undefined = -1
    OK = 0          # The island is solved without overflow
    OverFlow = 1    # The island is solved but overflow
    Failed = 2      # No island is solved

class Island:
    @staticmethod
    def from_grid(g: Grid) -> 'list[Island]':
        """Create a list of islands from the grid. An island is a set of buses that are connected to each other."""
        q: 'deque[BusID]' = deque()
        visited: 'set[BusID]' = set()
        islands: 'list[set[BusID]]' = []
        for bus in g.BusNames:
            if bus in visited: continue
            q.append(bus)
            island: 'set[BusID]' = set()
            while len(q) > 0:
                b = q.popleft()
                if b in visited: continue
                visited.add(b)
                island.add(b)
                for line in g.LinesOfFBus(b):
                    if line.tBus in visited: continue
                    q.append(line._tBus)
                for line in g.LinesOfTBus(b):
                    if line.fBus in visited: continue
                    q.append(line.fBus)
            islands.append(island)
        return [Island(g, island) for island in islands]
    
    def __init__(self, g: Grid, buses: Iterable[BusID]):
        self.grid = g
        self.Buses: 'set[str]' = set()
        self.Gens: 'set[str]' = set()
        self.Lines: 'set[str]' = set()
        self.PVWs: 'set[str]' = set()
        self.ESSs: 'set[str]' = set()
        self.result: IslandResult = IslandResult.Undefined
        self.result_value:float = 0.0
        for b in buses:
            self.Buses.add(b)
            self.Gens.update(gen.ID for gen in g.GensAtBus(b))
            self.Lines.update(ln.ID for ln in g.LinesOfFBus(b))
            self.Lines.update(ln.ID for ln in g.LinesOfTBus(b))
            self.PVWs.update(p.ID for p in g._patb[b])
            self.ESSs.update(e.ID for e in g._eatb[b])
    
    def __repr__(self):
        return f"Island: {self.Buses}\nGens: {self.Gens}\nLines: {self.Lines}\nPVWs: {self.PVWs}\nESSs: {self.ESSs}"
    
    def __str__(self):
        return self.__repr__()
    
    def BusItems(self):
        for b in self.Buses:
            yield b, self.grid.Bus(b)
    
    def GenItems(self):
        for g in self.Gens:
            yield g, self.grid.Gen(g)
    
    def LineItems(self):
        for l in self.Lines:
            yield l, self.grid.Line(l)
    
    def PVWItems(self):
        for p in self.PVWs:
            yield p, self.grid.PVWind(p)
    
    def ESSItems(self):
        for e in self.ESSs:
            yield e, self.grid.ESS(e)
    
    def YMat(self) -> 'tuple[dict[str,int], np.ndarray]':
        '''Return the admittance matrix of the grid'''
        n = len(self.Buses)
        bus_dict = {b: i for i, b in enumerate(self.Buses)}
        Y = np.zeros((n, n), dtype=complex)
        for _, l in self.LineItems():
            i = bus_dict[l.fBus]
            j = bus_dict[l.tBus]
            y = 1 / l.Z
            Y[i, j] -= y
            Y[j, i] -= y
            Y[i, i] += y
            Y[j, j] += y
        return bus_dict, Y