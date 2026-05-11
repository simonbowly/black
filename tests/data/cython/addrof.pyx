def addrof_name(double x):
    cdef double *p = &x
    return p[0]+1.0


def addrof_paren(double x):
    cdef double *q = &(x)
    return q[0]+2.0


def addrof_attr(object obj):
    cdef void *p = &obj.value
    return p
