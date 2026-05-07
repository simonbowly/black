cdef struct Point:
    double x
    double y


cdef union Data:
    int i
    double d


ctypedef struct Color:
    int r
    int g
    int b


def distance(Point p1, Point p2):
    return ((p1.x-p2.x)**2+(p1.y-p2.y)**2)**0.5
