import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atom_solver import solve_atom
from data_classes import IterationData

class TestAtomSolver(unittest.TestCase):
    def test_solve_atom_helium(self):
        # Test for Z=2 (Helium)
        iteration_data = solve_atom(Z=2, R=20, K=500, max_iterations=10, tol=1e-6)
        self.assertIsInstance(iteration_data, list)
        self.assertIsInstance(iteration_data[0], IterationData)
        self.assertGreater(len(iteration_data), 1)
        
        final_states = iteration_data[-1].states
        self.assertAlmostEqual(final_states.energies[0], -0.9, places=1)


if __name__ == '__main__':
    unittest.main()
