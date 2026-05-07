cdef class Point:
    cdef int x
    cdef int y

    def method(self):
        return self.x + self.y


cdef class Line(object):
    cdef Point start
    cdef Point end

    def length(self):
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return (dx * dx + dy * dy) ** 0.5
