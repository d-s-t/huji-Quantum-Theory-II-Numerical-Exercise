import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multi_electron_solver import get_states, get_lower_energy_states

class TestMultiElectronSolver(unittest.TestCase):
    def test_get_states(self):
        # Test for Z=2 (Helium)
        states_He = get_states(2)
        expected_states_He = np.array([[1, 0, 0, -1],
                                       [1, 0, 0, 1]])
        np.testing.assert_array_equal(states_He, expected_states_He)

        # Test for Z=3 (Lithium)
        states_Li = get_states(3)
        expected_states_Li = np.array([[1, 0, 0, -1],
                                       [1, 0, 0, 1],
                                       [2, 0, 0, -1]]) # The third state should be the next available, which is 2s (n=2, l=0, m=0, s=-1)
        np.testing.assert_array_equal(states_Li, expected_states_Li)

        # test for Z=8 (Oxygen)
        states_O = get_states(8)
        expected_states_O = np.array([[1, 0, 0, -1],
                                      [1, 0, 0, 1],
                                      [2, 0, 0, -1],
                                      [2, 0, 0, 1],
                                      [2, 1, -1, -1],
                                      [2, 1, -1, 1],
                                      [2, 1, 0, -1],
                                      [2, 1, 0, 1]])
        np.testing.assert_array_equal(states_O, expected_states_O)



    def test_get_lower_energy_states_hydrogen(self):
        K = 500
        Vee = 0

        # Test for Z=1 (Hydrogen)
        Z_H = 1
        R=50
        evals_H, evecs_H = get_lower_energy_states(Z_H, R, K, Vee, return_eigenval=True)
        self.assertEqual(evals_H.shape, (Z_H,))
        self.assertEqual(evecs_H.shape, (Z_H, K))
        self.assertTrue(np.all(np.diff(evals_H) >= 0)) # Check if eigenvalues are sorted
        self.assertAlmostEqual(evals_H[0], -0.5, places=2) # Ground state energy for H

        # Test for Z=2 (Helium) - without electron-electron interaction
        Z_He = 2
        R=20
        evals_He, evecs_He = get_lower_energy_states(Z_He, R, K, Vee, return_eigenval=True)
        self.assertEqual(evals_He.shape, (Z_He,))
        self.assertEqual(evecs_He.shape, (Z_He, K))
        self.assertTrue(np.all(np.diff(evals_He) >= 0)) # Check if eigenvalues are sorted
        # For Z=2 without Vee, the energies should be -Z^2/(2n^2)
        # n=1, l=0, m=0, s=-1 -> E = -2^2/(2*1^2) = -2.0
        # n=1, l=0, m=0, s=1 -> E = -2^2/(2*1^2) = -2.0
        self.assertAlmostEqual(evals_He[0], -2.0, places=2)
        self.assertAlmostEqual(evals_He[1], -2.0, places=2)

        # Test for Z=8 (Oxygen) - without electron-electron interaction
        Z_O = 8
        R=10
        K=3000
        evals_O, evecs_O = get_lower_energy_states(Z_O, R, K, Vee, return_eigenval=True)
        self.assertEqual(evals_O.shape, (Z_O,))
        self.assertEqual(evecs_O.shape, (Z_O, K))
        self.assertTrue(np.all(np.diff(evals_O) >= 0)) # Check if eigenvalues are sorted
        # For Z=8 without Vee, the energies should be -Z^2/(2n^2)
        # n=1 states (2 of them) -> E = -8^2/(2*1^2) = -32.0
        # n=2 states (6 of them) -> E = -8^2/(2*2^2) = -8.0
        self.assertAlmostEqual(evals_O[0], -32.0, places=2)
        self.assertAlmostEqual(evals_O[1], -32.0, places=2)
        self.assertAlmostEqual(evals_O[2], -8.0, places=2)
        # self.assertAlmostEqual(evals_O[7], -8.0, places=2)
        

if __name__ == '__main__':
    unittest.main()