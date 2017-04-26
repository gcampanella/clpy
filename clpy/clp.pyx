import numpy as np
cimport numpy as np


INFINITY = DBL_MAX


cdef class Clp:
    def __cinit__(self):
        self._model = Clp_newModel()
        if self._model is NULL:
            raise MemoryError

    def __init__(self, np.ndarray[np.double_t, ndim=2, mode='fortran'] x,
                 np.ndarray[np.double_t, ndim=1] col_lb, np.ndarray[np.double_t, ndim=1] col_ub,
                 np.ndarray[np.double_t, ndim=1] obj,
                 np.ndarray[np.double_t, ndim=1] row_lb, np.ndarray[np.double_t, ndim=1] row_ub):
        self._n_constraints = x.shape[0]
        self._n_variables = x.shape[1]
        cdef np.ndarray[np.int32_t] col_starts = np.arange(self._n_variables + 1, dtype=np.int32) * self._n_constraints
        cdef np.ndarray[np.int32_t] row_indices = np.tile(np.arange(self._n_constraints, dtype=np.int32),
                                                          self._n_variables)
        Clp_loadProblem(self._model, self._n_variables, self._n_constraints,
                        &col_starts[0], &row_indices[0], &x[0, 0],
                        &col_lb[0], &col_ub[0], &obj[0], &row_lb[0], &row_ub[0])

    def __dealloc__(self):
        if self._model is not NULL:
            Clp_deleteModel(self._model)

    cpdef void initial_solve(self):
        Clp_initialDualSolve(self._model)

    cpdef double dual_tolerance(self):
        return Clp_dualTolerance(self._model)

    cpdef void set_dual_tolerance(self, double tolerance):
        Clp_setDualTolerance(self._model, tolerance)

    cpdef LogLevel log_level(self):
        return LogLevel(Clp_logLevel(self._model))

    cpdef void set_log_level(self, LogLevel level):
        Clp_setLogLevel(self._model, level)

    cpdef int max_iterations(self):
        return maximumIterations(self._model)

    cpdef void set_max_iterations(self, int n_iterations):
        Clp_setMaximumIterations(self._model, n_iterations)

    cpdef double objective_value(self):
        return Clp_objectiveValue(self._model)

    cpdef OptimizationDirection optimization_direction(self):
        return OptimizationDirection(Clp_optimizationDirection(self._model))

    cpdef void set_optimization_direction(self, OptimizationDirection direction):
        Clp_setOptimizationDirection(self._model, <double> direction)

    cpdef double primal_tolerance(self):
        return Clp_primalTolerance(self._model)

    cpdef void set_primal_tolerance(self, double tolerance):
        Clp_setPrimalTolerance(self._model, tolerance)

    cpdef np.ndarray[np.double_t, ndim=1] reduced_costs(self):
        cdef np.double_t[:] result = <np.double_t[:self._n_variables]> Clp_dualColumnSolution(self._model)
        return np.array(result)

    cpdef np.ndarray[np.double_t, ndim=1] row_activities(self):
        cdef np.double_t[:] result = <np.double_t[:self._n_constraints]> Clp_primalRowSolution(self._model)
        return np.array(result)

    cpdef Scaling scaling(self):
        return Scaling(Clp_scalingFlag(self._model))

    cpdef void set_scaling(self, Scaling mode):
        Clp_scaling(self._model, mode)

    cpdef np.ndarray[np.double_t, ndim=1] shadow_prices(self):
        cdef np.double_t[:] result = <np.double_t[:self._n_constraints]> Clp_dualRowSolution(self._model)
        return np.array(result)

    cpdef np.ndarray[np.double_t, ndim=1] solution(self):
        cdef np.double_t[:] result = <np.double_t[:self._n_variables]> Clp_primalColumnSolution(self._model)
        return np.array(result)

    cpdef Status status(self):
        return Status(Clp_status(self._model))
