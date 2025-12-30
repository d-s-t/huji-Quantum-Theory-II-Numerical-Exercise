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

