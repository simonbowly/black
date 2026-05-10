cdef int add(int a, int b) except -1:
    return a+b


cdef double norm() except *:
    return 1.0


cdef int maybe(int n) except ?-1:
    return n+1


cdef int acquire(int n) except -1 nogil:
    return n+1


cdef void trigger() except * with gil:
    pass
