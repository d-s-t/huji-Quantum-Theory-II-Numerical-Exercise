import numpy as np

def harmonic_oscillator_potential(r: np.ndarray) -> np.ndarray:
    """
    Example potential: Harmonic oscillator potential
    V(r) = r^2 / 2
    r: np.ndarray[float]
        Array of distances in fm
        shape: (N,)
    """
    return r**2/2


def hydrogen_atom_potential(r: np.ndarray) -> np.ndarray:
    """
    Example potential: Hydrogen atom potential
    V(r) = -1 / r
    r: np.ndarray[float]
        Array of distances in fm
        shape: (N,)
    """
    return -1 / r

def helium_atom_potential(r: np.ndarray) -> np.ndarray:
    """
    Example potential: Helium atom potential
    V(r) = -2 / r
    r: np.ndarray[float]
        Array of distances in fm
        shape: (N,)
    """
    return -2 / r