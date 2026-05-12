cdef extern from "pyx_utils.h":
    cdef int check_version "PyxCheckVersion"(int major,int minor)
    cdef void raise_error "PyxRaiseError"(int code)
    cdef double compute "PyxCompute"(double x,double y)


x = {"a": 1}
