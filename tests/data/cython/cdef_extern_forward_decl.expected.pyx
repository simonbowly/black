cdef extern from "math.h" nogil:
    cpdef double sqrt(double x)
    cdef double log(double x)
    double pow(double x, double y)


cdef extern from "string.h":
    int strlen(char * s)


x = sqrt(4.0) + log(2.0)
