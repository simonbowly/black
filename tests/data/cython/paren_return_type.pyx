cdef (int, double) make_pair(int x):
    return x, x*1.5

cpdef (char*) get_name(int idx):
    return b"item"

cdef (int, double) pair_var = (1, 1.5)
