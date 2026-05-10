cdef extern from "Python.h":
    ctypedef class __builtin__.list [object PyListObject]:
        cdef Py_ssize_t ob_size

    ctypedef class __builtin__.dict  [object PyDictObject]:
        pass

    ctypedef struct PyObject

    ctypedef class __builtin__.tuple [object PyTupleObject]:
        pass


cdef extern from "exceptions.h":
    ctypedef extern class builtins.Exception [object PyBaseExceptionObject]:
        pass


x = [1,2,3]
y = {'a':1}
