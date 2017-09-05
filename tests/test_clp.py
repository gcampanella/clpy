import numpy as np
import unittest

import clpy


class TestClp(unittest.TestCase):
    def test_solve(self):
        solver = self._example()
        solver.initial_solve()
        np.testing.assert_equal(solver.status(), int(clpy.Status.OPTIMAL))
        np.testing.assert_almost_equal(solver.objective_value(), 268.0)
        np.testing.assert_almost_equal(solver.solution(), [1.8, 20.8, 1.6])
        np.testing.assert_almost_equal(solver.reduced_costs(), np.zeros(3))
        np.testing.assert_almost_equal(solver.shadow_prices(), [1, 6, 0, 1])

    @staticmethod
    def _example() -> clpy.Clp:
        x = np.array([
            [3, 2, 5],
            [2, 1, 1],
            [1, 1, 3],
            [5, 2, 4]
        ], dtype=np.double, order='F')
        col_lb = np.zeros(3, dtype=np.double)
        col_ub = np.repeat(clpy.INFINITY, 3)
        obj = np.array([20, 10, 15], dtype=np.double)
        row_lb = np.repeat(-clpy.INFINITY, 4)
        row_ub = np.array([55, 26, 30, 57], dtype=np.double)
        solver = clpy.Clp(x, col_lb, col_ub, obj, row_lb, row_ub)
        solver.set_log_level(clpy.LogLevel.NONE)
        solver.set_optimization_direction(clpy.OptimizationDirection.MAXIMIZE)
        return solver
