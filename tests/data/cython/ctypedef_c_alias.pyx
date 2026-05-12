cdef extern from *:
    ctypedef long long int128_t "__int128_t"
    ctypedef unsigned long long uint128_t "__uint128_t"

cdef extern from "enum.h":
    cpdef enum MyEnum:
        ONE "1"
        TWO "2"

x = {'a':1}
