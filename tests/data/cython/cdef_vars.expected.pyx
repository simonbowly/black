cdef int x = 1
cdef double pi = 3.14159265358979
cdef bint flag = True


def foo():
    cdef int n = 10
    cdef double result
    result = n * pi
    return result
