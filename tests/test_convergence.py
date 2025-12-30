import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eigen_state_solvers import finite_difference_method_radial, numerov_method_radial

class TestConvergence(unittest.TestCase):
    def setUp(self):
        from potential_functions import harmonic_oscillator_potential
        self.V = harmonic_oscillator_potential
        self.R = 10.0  # fm
        self.l = 0  # azimuthal quantum number
        self.N_values = [40, 80, 120, 240, 360, 480, 600]
        self.exact_energy = 1.5  # MeV
        # Fit to a power law: error = C * N^{-q}
        self.power_law = lambda N, C, q: C * N**(-q)

    def test_finite_difference_convergence_rate(self):
        from scipy.optimize import curve_fit
        errors_fd = []
        for N in self.N_values:
            r = np.linspace(0, self.R, N+1)[1:]
            evals_fd, _ = finite_difference_method_radial(self.l, self.V, r)
            errors_fd.append(abs(evals_fd[0] - self.exact_energy))
        
        errors_fd = np.array(errors_fd)

        popt_fd, _ = curve_fit(self.power_law, self.N_values, errors_fd)

        # Check that the convergence rates are reasonable
        self.assertGreater(popt_fd[1], 2.0) 

    def test_numerov_convergence_rate(self):
        from scipy.optimize import curve_fit
        errors_nm = []
        for N in self.N_values:
            r = np.linspace(0, self.R, N+1)[1:]
            evals_nm, _ = numerov_method_radial(self.l, self.V, r)
            errors_nm.append(abs(evals_nm[0] - self.exact_energy))
        
        errors_nm = np.array(errors_nm)

        popt_nm, _ = curve_fit(self.power_law, self.N_values, errors_nm)

        # Check that the convergence rates are reasonable
        self.assertGreater(popt_nm[1], 4.0) 

if __name__ == '__main__':
    unittest.main()