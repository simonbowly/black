def process(int n):
    cdef int i, j, k
    cdef double total, factor = 2.0
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += i + j * factor
    return total
