from libc.stdint cimport int32_t

cdef extern from 'coin/Clp_C_Interface.h' nogil:
    ctypedef void *Clp_Simplex

    Clp_Simplex *Clp_newModel()
    void Clp_deleteModel(Clp_Simplex *model)

    void Clp_loadProblem(Clp_Simplex *model, int n_cols, int n_rows, int32_t *col_starts, int32_t *row_indices,
                         double *x, double *col_lb, double *col_ub, double *obj, double *row_lb, double *row_ub)

    double Clp_primalTolerance(Clp_Simplex *model)
    void Clp_setPrimalTolerance(Clp_Simplex *model, double tolerance)

    double Clp_dualTolerance(Clp_Simplex *model)
    void Clp_setDualTolerance(Clp_Simplex *model, double value)

    int Clp_logLevel(Clp_Simplex *model)
    void Clp_setLogLevel(Clp_Simplex *model, int level)

    int maximumIterations(Clp_Simplex *model)  # No Clp_ prefix
    void Clp_setMaximumIterations(Clp_Simplex *model, int n_iterations)

    double Clp_optimizationDirection(Clp_Simplex *model)
    void Clp_setOptimizationDirection(Clp_Simplex *model, double direction)

    int Clp_scalingFlag(Clp_Simplex *model)
    void Clp_scaling(Clp_Simplex *model, int mode)

    int Clp_primal(Clp_Simplex *model, int mode)
    int Clp_initialPrimalSolve(Clp_Simplex *model)

    int Clp_dual(Clp_Simplex *model, int mode)
    int Clp_initialDualSolve(Clp_Simplex *model)

    int Clp_status(Clp_Simplex *model)
    int Clp_getColumnStatus(Clp_Simplex *model, int index)
    int Clp_getRowStatus(Clp_Simplex *model, int index)

    double Clp_objectiveValue(Clp_Simplex *model)

    double *Clp_primalRowSolution(Clp_Simplex *model)
    double *Clp_primalColumnSolution(Clp_Simplex *model)
    double *Clp_dualRowSolution(Clp_Simplex *model)
    double *Clp_dualColumnSolution(Clp_Simplex *model)
