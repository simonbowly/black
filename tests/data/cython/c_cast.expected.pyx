def cast_demo(int x, double y):
    cdef int i = <int>y
    cdef double d = <double>(x + 1)
    return i + d


def cast_bytes(char *s):
    return <bytes>s


def cast_multi():
    cdef signed char sc = <signed char>((<unsigned char>-1) >> 1)
    return sc


def cast_string_char():
    cdef char c = <char>'A'
    return c
