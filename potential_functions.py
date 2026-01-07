import numpy as np
from typing import Callable

def harmonic_oscillator_potential(r: np.ndarray) -> np.ndarray:
    """
    Example potential: Harmonic oscillator potential
    V(r) = r^2 / 2
    r: np.ndarray[float]
        Array of distances in fm
        shape: (N,)
    """
    return r**2/2


def coulomb_potential(Z: int) -> Callable[[np.ndarray], np.ndarray]:
    """
    Example potential: Coulomb potential
    V(r) = -Z / r
    Z: int
        Atomic number
    returns: Callable[[np.ndarray[float]], np.ndarray[float]]
        Potential energy function
    """
    return lambda r: -Z / r

hydrogen_atom_potential = coulomb_potential(1)
