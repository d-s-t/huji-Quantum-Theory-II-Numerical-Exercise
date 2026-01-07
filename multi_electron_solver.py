from eigen_state_solvers import numerov_method_coulomb_l0_2nd_order as l0_solver, numerov_method_coulomb_l1_2nd_order as l1_solver
import numpy as np
from typing import Callable
from potential_functions import coulomb_potential
from dataclasses import dataclass

def get_states(Z: int) -> np.ndarray:
    """
    Get the |nlms> states needed for Z electrons in the atom.

    :param Z: Atomic number
    :type Z: int
    :return: Array of shape (Z,4) with each row being (n,l,m,s)
    :rtype: ndarray
    """
    return np.array([(n, l, m, s)
                     for n in range(1, Z + 1)
                     for l in range(n)
                     for m in range(-l, l + 1)
                     for s in [-1, 1]
                     ], dtype=int)[:Z]

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
        return r"$|{n=}, {l=}, {m=}, \sigma={s}\rangle$".format(n=self.n, l=self.l, m=self.m, s=self.s)

@dataclass
class NLMS_States:
    evals: list[np.ndarray]
    evecs: list[np.ndarray]
    Z: int

    def __getitem__(self, state: tuple[int, int, int, int]) -> np.ndarray:
        n, l, _, _ = state
        idx = n + l - 1
        return self.evecs[l][:, idx]
    
    def energy(self, state: tuple[int, int, int, int]) -> float:
        n, l, _, _ = state
        idx = n + l - 1
        return self.evals[l][idx]
    
    # implement state in nlms_states
    def __iter__(self):
        return iter(get_states(self.Z))
    
    def __len__(self):
        return len(self.Z)



def get_lower_energy_states(Z: int, R: float, K: int, Vee: Callable[[np.ndarray], np.ndarray]) -> NLMS_States:
    """
    Get the Z lowest energy states for Z atomic number, with colomb potential and electron-electron interaction Vee.
    if Z<=2, use only l=0 solver, else use l=0 and l=1 solvers to get enough states.
    
    :param Z: Description
    :type Z: int
    :param R: Description
    :type R: float
    :param K: Description
    :type K: int
    :param Vee: Description
    :type Vee: Callable[[np.ndarray], np.ndarray]
    :return: Description
    :rtype: ndarray
    """
    if Z <= 0:
        raise ValueError("Z must be a positive integer")
    if Z>10:
        raise NotImplementedError("This function only supports up to Z=10")
    Vcol = coulomb_potential(Z)
    V = lambda r: Vcol(r) + Vee
    evals0, evecs0 = l0_solver(Z, R, K, V)
    evals1, evecs1 = l1_solver(Z, R, K, V)
    all_evals = [evals0, evals1]
    all_evecs = [evecs0, evecs1]
    return NLMS_States(all_evals, all_evecs, Z=Z)

def get_electron_density(states: NLMS_States, r: np.ndarray, Z: int) -> np.ndarray:
    """
    Compute the electron density ρ(r) from the given eigenstates.

    :param states: Eigenstates of shape (Z, K)
    :type states: NLMS_States
    :param r: Radial grid points of shape (K,)
    :type r: ndarray
    :param Z: Atomic number
    :type Z: int
    :return: Electron density ρ(r) of shape (K,)
    :rtype: ndarray
    """
    s = get_states(Z)
    K = r.shape[0]
    rho = np.zeros(K, dtype=np.float64)
    for state in s:
        evec = states[state]/r
        rho += np.abs(evec)**2
    return rho



def get_electron_electron_potential(prev_Vee: np.ndarray, rho: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Compute the electron-electron interaction potential Vee(r) from the given eigenstates.
    V_{ee}^{(n+1)} = \\frac{1}{2} \\left[ V_{ee}^{(n)} + \\int d\\boldsymbol{r}' \\frac{\\rho_{e}^{(n)}(r')}{|\\boldsymbol{r}' - \\boldsymbol{r}|} \\right]

    :param rho: Electron density of shape (K,)
    :type rho: ndarray
    :param r: Radial grid points of shape (K,)
    :type r: ndarray
    :return: Electron-electron interaction potential Vee(r) of shape (K,)
    :rtype: ndarray
    """
    K = r.shape[0]

    numerator = rho * r**2
    denominator = np.abs(r[:, None] - r[None, :])
    integrand = numerator / denominator
    integrand[np.eye(K)==1] = 0 
    Vee_new = 4 * np.pi * np.trapezoid(integrand, r, axis=1)

    return 0.5 * (prev_Vee + Vee_new)

@dataclass
class IterationData:
    Vee: np.ndarray
    states: NLMS_States
    rho: np.ndarray

def solve_multi_electron_atom(Z: int, R: float, K: int, max_iterations: int = 100, tol: float = 1e-6) -> list[IterationData]:
    """
    Solve the multi-electron atom problem using self-consistent field method.

    :param Z: Atomic number
    :type Z: int
    :param R: Maximum distance
    :type R: float
    :param K: Number of points in the grid
    :type K: int
    :param max_iterations: Maximum number of iterations
    :type max_iterations: int
    :param tol: Tolerance for convergence
    :type tol: float
    :return: Eigenstates of the multi-electron atom
    :rtype: NLMS_States
    """
    r = np.linspace(0, R, K+1)[1:]
    Vee = np.zeros(K, dtype=np.float64)
    iteration_data = []

    for i in range(max_iterations):
        states = get_lower_energy_states(Z, R, K, Vee)
        rho = get_electron_density(states, r, Z)
        iteration_data.append(IterationData(Vee=Vee.copy(), states=states, rho=rho.copy()))
        Vee_new = get_electron_electron_potential(Vee, rho, r)

        if np.linalg.norm(Vee_new - Vee) < tol:
            break

        Vee = Vee_new

    return iteration_data