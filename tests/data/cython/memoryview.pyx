def process(double[:] arr, int n):
    return arr[0]+n


cdef void fill(double[:, :] matrix, int rows, int cols):
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = 0.0


cdef double[:] buf
cdef int[:, :] grid
