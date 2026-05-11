ctypedef int[:, ::1] memview_int
ctypedef double[:] memview_double

def process(memview_int a, memview_double b):
    return a[0, 0] + b[0]
