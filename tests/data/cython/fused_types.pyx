ctypedef fused number:
    int
    float
    double


cdef fused integer_t:
    int
    long


cdef double process(number x):
    return x+1.0
