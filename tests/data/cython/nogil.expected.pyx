cdef int add(int a, int b) nogil:
    return a + b


cdef void clear(double[:] arr) nogil:
    for i in range(arr.shape[0]):
        arr[i] = 0.0


cpdef double scale(double x, double factor) nogil:
    return x * factor


cdef int acquire(int n) with gil:
    return n + 1
