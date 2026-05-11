cdef void demo(char* buf):
    buf[0] = c'\0'
    buf[1] = c'A'
    buf[2] = c'\n'
    if buf[0] == c'x':
        buf[0] = c'\x10'
    result = buf[0] + buf[1]
