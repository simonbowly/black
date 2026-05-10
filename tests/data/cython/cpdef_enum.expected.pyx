cpdef enum Color:
    RED = 0
    GREEN = 1
    BLUE = 2


cpdef enum:
    SMALL = 1
    BIG = 2


def blend(Color a, Color b):
    return a + b
