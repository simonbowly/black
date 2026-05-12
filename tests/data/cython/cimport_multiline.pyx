from cpython.contextvars cimport (
    PyContextVar_New,
    PyContextVar_New_with_default,
    get_value,
    get_value_no_default,
)
from libcpp.algorithm cimport (sort,stable_sort,nth_element)

x = {'a':1,'b':2}
