import numpy as np
cimport numpy as np

from .clp_interface cimport *

cdef extern from '<float.h>' nogil:
    double DBL_MAX
    double DBL_MIN

cpdef enum LogLevel:
    NONE = 0
    FINAL = 1
    FACTORIZATIONS = 2
    VERBOSE = 4

cpdef enum OptimizationDirection:
    IGNORE = 0
    MAXIMIZE = -1
    MINIMIZE = 1

cpdef enum Scaling:
    OFF = 0
    EQUILIBRIUM = 1
    GEOMETRIC = 2
    AUTO = 3
    DYNAMIC = 4

cpdef enum Status:
    OPTIMAL = 0
    PRIMAL_INFEASIBLE = 1
    DUAL_INFEASIBLE = 2
    STOPPED_ON_ITERATIONS = 3
    STOPPED_DUE_TO_ERRORS = 4

cdef class Clp:
    cdef Clp_Simplex *_model
    cdef int _n_constraints
    cdef int _n_variables

    cpdef void initial_solve(self)

    cpdef double dual_tolerance(self)
    cpdef void set_dual_tolerance(self, double tolerance)

    cpdef LogLevel log_level(self)
    cpdef void set_log_level(self, LogLevel level)

    cpdef int max_iterations(self)
    cpdef void set_max_iterations(self, int n_iterations)

    cpdef double objective_value(self)

    cpdef OptimizationDirection optimization_direction(self)
    cpdef void set_optimization_direction(self, OptimizationDirection direction)

    cpdef double primal_tolerance(self)
    cpdef void set_primal_tolerance(self, double tolerance)

    cpdef np.ndarray[np.double_t, ndim=1] reduced_costs(self)

    cpdef np.ndarray[np.double_t, ndim=1] row_activities(self)

    cpdef Scaling scaling(self)
    cpdef void set_scaling(self, Scaling mode)

    cpdef np.ndarray[np.double_t, ndim=1] shadow_prices(self)

    cpdef np.ndarray[np.double_t, ndim=1] solution(self)

    cpdef Status status(self)
