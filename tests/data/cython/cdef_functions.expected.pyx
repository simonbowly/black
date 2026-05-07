cdef int add(int a, int b):
    return a + b


cdef double square_root(double x):
    return x * x


cpdef int public_add(int a, int b):
    return a + b


def python_func(x, y):
    return x + y
