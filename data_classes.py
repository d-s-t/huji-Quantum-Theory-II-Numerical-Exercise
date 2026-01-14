from dataclasses import dataclass
from typing import Iterator
import numpy as np


@dataclass
class NLMS_State:
    n: int
    l: int
    m: int
    s: int

    # define unpack method
    def __iter__(self):
        return iter((self.n, self.l, self.m, self.s))
    
    def __str__(self):
        return r"|n={n}, l={l}, m={m}, \sigma={s}\rangle".format(n=self.n, l=self.l, m=self.m, s=self.s)

@dataclass
class NLMS_States:
    evals: list[np.ndarray]
    evecs: list[np.ndarray]
    Z: int

    def __getitem__(self, state: NLMS_State) -> np.ndarray:
        idx = state.n - state.l - 1
        return self.evecs[state.l][:, idx]
    
    def energy(self, state: NLMS_State) -> float:
        idx = state.n - state.l - 1
        return self.evals[state.l][idx]
    
    def __iter__(self) -> Iterator[NLMS_State]:
        from itertools import islice
        return islice((NLMS_State(n, l, m, s)
                        for n in range(1, self.Z + 1)
                        for l in range(n)
                        for m in range(-l, l + 1)
                        for s in [-1, 1]
                        ), self.Z)
    
    def __len__(self):
        return self.Z
    

@dataclass
class IterationData:
    Vee: np.ndarray
    states: NLMS_States
    rho: np.ndarray
