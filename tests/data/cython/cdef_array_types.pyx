cdef struct StructWithArray:
    int data[4]
    MyStruct[2] b

cdef void f(int x[2][2]):
    pass

cdef int arr[MY_SIZE]

x = {'a':1}
