cdef extern from *:
    ctypedef long ssize_t
    double sqrt(double x)
    int RAND_MAX

y = sqrt( RAND_MAX )
