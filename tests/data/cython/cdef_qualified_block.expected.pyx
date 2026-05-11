cdef class Container:
    cdef readonly:
        double x

    def value(self):
        return self.x * 2.0
