from .clp_interface cimport *

cdef extern from '<float.h>' nogil:
    double DBL_MAX

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
    FREE = 0
    BASIC = 1
    AT_UPPER_BOUND = 2
    AT_LOWER_BOUND = 3
    SUPER_BASIC = 4
    FIXED = 5

cpdef enum ProblemStatus:
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

    cpdef double[:] reduced_costs(self)

    cpdef double[:] row_activities(self)

    cpdef Scaling scaling(self)
    cpdef void set_scaling(self, Scaling mode)

    cpdef double[:] shadow_prices(self)

    cpdef double[:] solution(self)

    cpdef ProblemStatus status(self)
    cpdef Status column_status(self, int index)
    cpdef Status row_status(self, int index)
