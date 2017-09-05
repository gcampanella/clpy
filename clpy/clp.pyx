from cython.view cimport array as cvarray
from libc.stdint cimport int32_t

INFINITY = DBL_MAX


cdef class Clp:
    def __cinit__(self):
        self._model = Clp_newModel()
        if self._model is NULL:
            raise MemoryError

    def __init__(self, double[::1, :] x,
                 double[:] col_lb, double[:] col_ub,
                 double[:] obj,
                 double[:] row_lb, double[:] row_ub):
        self._n_constraints = x.shape[0]
        self._n_variables = x.shape[1]
        cdef int32_t[:] col_starts = compute_col_starts(self._n_constraints, self._n_variables)
        cdef int32_t[:] row_indices = compute_row_indices(self._n_constraints, self._n_variables)
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

    cpdef double[:] reduced_costs(self):
        return <double[:self._n_variables]> Clp_dualColumnSolution(self._model)

    cpdef double[:] row_activities(self):
        return <double[:self._n_constraints]> Clp_primalRowSolution(self._model)

    cpdef Scaling scaling(self):
        return Scaling(Clp_scalingFlag(self._model))

    cpdef void set_scaling(self, Scaling mode):
        Clp_scaling(self._model, mode)

    cpdef double[:] shadow_prices(self):
        return <double[:self._n_constraints]> Clp_dualRowSolution(self._model)

    cpdef double[:] solution(self):
        return <double[:self._n_variables]> Clp_primalColumnSolution(self._model)

    cpdef Status status(self):
        return Status(Clp_status(self._model))


cdef int32_t[:] compute_col_starts(int n_constraints, int n_variables):
    cdef int32_t[:] col_starts = cvarray(shape=(n_variables + 1,), itemsize=sizeof(int32_t), format='i')
    cdef int i
    for i in range(n_variables + 1):
        col_starts[i] = i * n_constraints
    return col_starts


cdef int32_t[:] compute_row_indices(int n_constraints, int n_variables):
    cdef int32_t[:] row_indices = cvarray(shape=(n_constraints * n_variables,), itemsize=sizeof(int32_t), format='i')
    cdef int i, j
    for i in range(n_variables):
        for j in range(n_constraints):
            row_indices[i * n_constraints + j] = j
    return row_indices
