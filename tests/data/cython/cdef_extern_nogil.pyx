cdef extern from "math.h" nogil:
    double sqrt(double x)
    double log(double x)

cdef extern from * namespace "utils" nogil:
    int flag

x = sqrt(4.0)
