cdef extern from "math.h" nogil:
    double sqrt(double x)


cdef extern from "string.h" nogil:
    int strlen(char *s)
    char *strcpy(char *dst, char *src)


cdef extern from * namespace "utils" nogil:
    int flag


x = sqrt(4.0)
