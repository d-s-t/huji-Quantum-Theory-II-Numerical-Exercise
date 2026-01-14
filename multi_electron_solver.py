from eigen_state_solvers import (
    numerov_method_coulomb_l0_2nd_order as l0_solver,
    numerov_method_coulomb_l1_2nd_order as l1_solver
)
import numpy as np
from potential_functions import coulomb_potential
from tqdm import trange
from data_classes import NLMS_States, IterationData



def get_lower_energy_states(Z: int, R: float, K: int, Vee: np.ndarray) -> NLMS_States:
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
    :type Vee: np.ndarray
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

def get_electron_density(states: NLMS_States) -> np.ndarray:
    """
    Compute the electron radial density \\lambda_e(r) from the given eigenstates.

    :param states: Eigenstates of shape (Z, K)
    :type states: NLMS_States
    :return: Electron radial density \\lambda_e(r) of shape (K,)
    :rtype: ndarray
    """
    return sum(np.abs(states[state])**2 for state in states)

def get_electron_electron_potential(prev_Vee: np.ndarray, electron_density: np.ndarray, r: np.ndarray, Z: int) -> np.ndarray:
    """
    Compute the electron-electron interaction potential Vee(r) from the given eigenstates.
    V_{ee}(r) = 4\\pi\\frac{Z-1}{Z}\\int_0^\\infty dr' \\frac{\\lambda_e(r')}{\\max(r,r')}

    :param electron_density: Electron radial density \\lambda_e(r) of shape (K,)
    :type electron_density: ndarray
    :param r: Radial grid points of shape (K,)
    :type r: ndarray
    :return: Electron-electron interaction potential Vee(r) of shape (K,)
    :rtype: ndarray
    """
    integrand = electron_density / np.maximum(r[:, None], r[None, :])
    Vee_new = (Z-1)/Z * np.trapezoid(integrand, r, axis=1)

    return 0.5 * (prev_Vee + Vee_new)

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


    with trange(100) as pbar:
        initial_dV = None
        pbar.set_description(f'multi-electron atom Z={Z}')
        pbar.set_postfix({'iter': f'0/{max_iterations}', 'ΔVee': None, 'tol': f"{tol:.2e}"})
        for i in range(max_iterations):
            states = get_lower_energy_states(Z, R, K, Vee)
            electron_density = get_electron_density(states)
            iteration_data.append(IterationData(Vee=Vee.copy(), states=states, electron_density=electron_density.copy()))
            Vee_new = get_electron_electron_potential(Vee, electron_density, r, Z)
            dV = np.linalg.norm(Vee_new - Vee)
            if initial_dV is None:
                initial_dV = dV
            convergence_percent = 100 * (np.log(dV / initial_dV) / np.log(tol / initial_dV)) if initial_dV != 0 else 0
            iteration_percent = 100 * i // max_iterations
            pbar.n = min(int(max(convergence_percent, iteration_percent)), 100)
            pbar.set_postfix({'iter': f'{i+1}/{max_iterations}', 'ΔVee': f"{dV:.2e}", 'tol': f"{tol:.2e}"})
            pbar.refresh()
            if dV < tol:
                return iteration_data
            Vee = Vee_new

    # print warning if not converged, use yelow color
    print("\033[93mWarning: Maximum iterations reached without convergence.\033[0m")
    return iteration_data