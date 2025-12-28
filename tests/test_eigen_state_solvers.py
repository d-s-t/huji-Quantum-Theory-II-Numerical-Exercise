import unittest
import numpy as np
from sys import float_info
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from eigen_state_solvers import finite_difference_method_radial, numerov_method_radial

class TestEigenStateSolvers(unittest.TestCase):
    def setUp(self):
        self.V = utils.harmonic_oscillator_potential
        self.R = 20.0  # fm
        self.K = 1000  # number of points
        self.l = 0  # azimuthal quantum number
        self.r = np.linspace(float_info.epsilon, self.R, self.K)
        self.evals_fd, self.evecs_fd = finite_difference_method_radial(self.l, self.V, self.r)
        self.evals_nm, self.evecs_nm = numerov_method_radial(self.l, self.V, self.r)

    def test_evals_ordered(self):
        self.assertTrue(np.all(np.diff(self.evals_fd) > 0))
        self.assertTrue(np.all(np.diff(self.evals_nm) > 0))

    def test_relative_error(self):
        rel_error = utils.relative_error(self.evals_fd, self.evals_nm)
        self.assertLess(max(rel_error[:self.K//10]), 1e-2) # Check that the relative error is small

    def test_wavefunction_normalization(self):
        norm_fd = np.trapezoid(self.evecs_fd * np.conjugate(self.evecs_fd), self.r, axis=0)
        norm_nm = np.trapezoid(self.evecs_nm * np.conjugate(self.evecs_nm), self.r, axis=0)
        self.assertTrue(np.allclose(norm_fd, np.ones_like(self.evals_fd)))
        self.assertTrue(np.allclose(norm_nm, np.ones_like(self.evals_nm)))

if __name__ == '__main__':
    unittest.main()
