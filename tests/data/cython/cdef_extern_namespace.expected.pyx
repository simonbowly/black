cdef extern from "shapes.h" namespace "shapes":
    double get_area(double r)
    double PI


cdef extern from * namespace "utils":
    int flag


x = get_area(PI)
