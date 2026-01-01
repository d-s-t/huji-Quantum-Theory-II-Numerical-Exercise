import numpy as np
from typing import Callable, Union
from scipy.linalg import eig, eigh_tridiagonal

def _W(V: Callable[[np.ndarray], np.ndarray], l: int, r: np.ndarray) -> np.ndarray:
    """
    W function for the radial equation

    V: Callable[[np.ndarray], np.ndarray]
        Potential energy function
    l: int
        Azimuthal quantum number
    r: np.ndarray[float]
        Array of distances in fm
        shape: (N,)
    """
    return V(r) + (l * (l + 1)) / (2 * r**2)

def finite_difference_method_radial(l: int, V: Callable[[np.ndarray], np.ndarray], r: np.ndarray, *, eigvals_only: bool = False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Finite difference method to solve the radial equation for the wave function

    l: int
        Azimuthal quantum number
    V: Callable[[np.ndarray], np.ndarray]
        Potential energy function
    r: np.ndarray[float]
        Array of distances
        shape: (N,)

    returns: tuple[np.ndarray[float], np.ndarray[float]]
        Eigenvalues and eigenvectors of the Hamiltonian
        shape: (N,), (N, N)
    """
    dr = r[1] - r[0]

    main_diag = 1/dr**2 + _W(V, l, r)
    off_diag = - np.ones(len(r) - 1) / (2*dr**2)

    if eigvals_only:
        evals = eigh_tridiagonal(main_diag, off_diag, eigvals_only=True)
        return evals
    
    evals, evecs = eigh_tridiagonal(main_diag, off_diag)
    normalized_evecs = evecs / np.sqrt(np.trapezoid(evecs * np.conjugate(evecs), r, axis=0))
    return evals, normalized_evecs


def numerov_method_radial(l: int, V: Callable[[np.ndarray], np.ndarray], r: np.ndarray, *, eigvals_only: bool = False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Numerov method to solve the radial equation for the wave function

    Note: This method assumes the potential is smooth. It may produce inaccurate
    results for potentials with singularities at r=0 (like Coulomb -1/r) due to
    the breakdown of the Taylor expansion and boundary term handling.

    l: int
        Azimuthal quantum number
    V: Callable[[np.ndarray], np.ndarray]
        Potential energy function
    r: np.ndarray[float]
        Array of distances
        shape: (N,)

    returns: tuple[np.ndarray[float], np.ndarray[float]]
        Eigenvalues and eigenvectors of the Hamiltonian
        shape: (N,), (N, N)
    """
    dr = r[1] - r[0]

    W = _W(V, l, r)
    H_main_diag = 1/(dr**2) + W * 5 / 6
    H_1up_diag = -1/(2*dr**2) + W[1:]/12
    H_1low_diag = -1/(2*dr**2) + W[:-1]/12
    H = np.diag(H_main_diag) + np.diag(H_1up_diag, 1) + np.diag(H_1low_diag, -1)

    N_main_diag = 5/6 * np.ones_like(r)
    N_off_diag = 1/12 * np.ones(len(r) - 1)
    N = np.diag(N_main_diag) + np.diag(N_off_diag, 1) + np.diag(N_off_diag, -1)

    if eigvals_only:
        evals = eig(H, N, right=False)
        return np.sort(evals.real)

    evals, evecs = eig(H, N)
    # sort eigenvalues (eig does not guarantee order) and take real part
    idx = evals.real.argsort()
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # return the real parts of the eigenvalues and eigenvectors
    normalized_evecs = evecs / np.sqrt(np.trapezoid(evecs * np.conjugate(evecs), r, axis=0))
    return evals.real, normalized_evecs


def numerov_method_coulomb_l0_1st_order(Z: int, R: float, K: int, *, eigvals_only: bool = False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Numerov method specialized for l=0 radial equation using first-order boundary condition at r=0.

    Z: int
        Atomic number
    R: float
        Maximum distance
    K: int
        Number of points in the grid
    eigvals_only: bool
        If True, only eigenvalues are returned
    """
    from potential_functions import coulomb_potential
    V = coulomb_potential(Z)
    dr = R / K
    r = np.linspace(dr, R, K)

    N_main_diag = np.full_like(r, 5/6)
    N_off_diag = np.full(len(r) - 1, 1/12)
    N = np.diag(N_main_diag) + np.diag(N_off_diag, 1) + np.diag(N_off_diag, -1)

    W = V(r)
    H_main_diag = 1/(dr**2) + W * 5 / 6
    H_off_diag = -1 / (2 * dr**2) + W/12
    H = np.diag(H_main_diag) + np.diag(H_off_diag[1:], 1) + np.diag(H_off_diag[:-1], -1)
    H[0,0] -= Z / (12*dr)

    if eigvals_only:
        evals = eig(H, N, right=False)
        return np.sort(evals.real)

    evals, evecs = eig(H, N)
    # sort eigenvalues (eig does not guarantee order) and take real part
    idx = evals.real.argsort()
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # return the real parts of the eigenvalues and eigenvectors
    normalized_evecs = evecs / np.sqrt(np.trapezoid(evecs * np.conjugate(evecs), r, axis=0))
    return evals.real, normalized_evecs


def numerov_method_coulomb_l0_2nd_order(Z: int, R: float, K: int, *, eigvals_only: bool = False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Numerov method specialized for l=0 radial equation using second-order boundary condition at r=0.

    Z: int
        Atomic number
    R: float
        Maximum distance
    K: int
        Number of points in the grid
    eigvals_only: bool
        If True, only eigenvalues are returned
    """
    from potential_functions import coulomb_potential
    V = coulomb_potential(Z)
    dr = R / K
    r = np.linspace(dr, R, K)

    N_main_diag = np.full_like(r, 5/6)
    N_off_diag = np.full(len(r) - 1, 1/12)
    N = np.diag(N_main_diag) + np.diag(N_off_diag, 1) + np.diag(N_off_diag, -1)

    W = V(r)
    H_main_diag = 1/(dr**2) + W * 5 / 6
    H_off_diag = -1 / (2 * dr**2) + W/12
    H = np.diag(H_main_diag) + np.diag(H_off_diag[1:], 1) + np.diag(H_off_diag[:-1], -1)
    H[0,0] += Z / (12*dr*(Z*dr - 1))

    if eigvals_only:
        evals = eig(H, N, right=False)
        return np.sort(evals.real)

    evals, evecs = eig(H, N)
    # sort eigenvalues (eig does not guarantee order) and take real part
    idx = evals.real.argsort()
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # return the real parts of the eigenvalues and eigenvectors
    normalized_evecs = evecs / np.sqrt(np.trapezoid(evecs * np.conjugate(evecs), r, axis=0))
    return evals.real, normalized_evecs