import numpy as np
from typing import Callable
from scipy.linalg import eig, eigh_tridiagonal

def _W(V: Callable[[np.ndarray], np.ndarray], l: int, r: np.ndarray) -> np.ndarray:
    """
    W function for the radial equation

    V: Callable[[np.ndarray[Quantity["fm"]]], np.ndarray[Quantity["MeV"]]]
        Potential energy function
    l: int
        Azimuthal quantum number
    r: np.ndarray[float]
        Array of distances in fm
        shape: (N,)
    """
    return V(r) + (l * (l + 1)) / (2 * r**2)

def finite_difference_method_radial(l: int, V: Callable[[np.ndarray], np.ndarray], r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite difference method to solve the radial equation for the wave function

    l: int
        Azimuthal quantum number
    V: Callable[[np.ndarray[Quantity["fm"]]], np.ndarray[Quantity["MeV"]]]
        Potential energy function
    r: np.ndarray[float]
        Array of distances in fm
        shape: (N,)

    returns: tuple[np.ndarray[float], np.ndarray[float]]
        Eigenvalues and eigenvectors of the Hamiltonian
        shape: (N,), (N, N)
    """
    dr = r[1] - r[0]

    main_diag = 1/dr**2 + _W(V, l, r)
    off_diag = - np.ones(len(r) - 1) / (2*dr**2)

    evals, evecs = eigh_tridiagonal(main_diag, off_diag)
    normalized_evecs = evecs / np.sqrt(np.trapezoid(evecs * np.conjugate(evecs), r, axis=0))
    return evals, normalized_evecs


def numerov_method_radial(l: int, V: Callable[[np.ndarray], np.ndarray], r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Numerov method to solve the radial equation for the wave function

    l: int
        Azimuthal quantum number
    V: Callable[[np.ndarray[Quantity["fm"]]], np.ndarray[Quantity["MeV"]]]
        Potential energy function
    r: np.ndarray[float]
        Array of distances in fm
        shape: (N,)

    returns: tuple[np.ndarray[float], np.ndarray[float]]
        Eigenvalues and eigenvectors of the Hamiltonian
        shape: (N,), (N, N)
    """
    dr = r[1] - r[0]

    W_values = _W(V, l, r)
    H_main_diag = 1/(dr**2) + W_values * 5 / 6
    H_1up_diag = -1/(2*dr**2) + W_values[1:]/12
    H_1low_diag = -1/(2*dr**2) + W_values[:-1]/12
    H = np.diag(H_main_diag) + np.diag(H_1up_diag, 1) + np.diag(H_1low_diag, -1)

    N_main_diag = 5/6 * np.ones_like(r)
    N_off_diag = 1/12 * np.ones(len(r) - 1)
    N = np.diag(N_main_diag) + np.diag(N_off_diag, 1) + np.diag(N_off_diag, -1)

    evals, evecs = eig(H, N)
    # sort eigenvalues (eig does not guarantee order) and take real part
    idx = evals.real.argsort()
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # return the real parts of the eigenvalues and eigenvectors
    normalized_evecs = evecs / np.sqrt(np.trapezoid(evecs * np.conjugate(evecs), r, axis=0))
    return evals.real, normalized_evecs
