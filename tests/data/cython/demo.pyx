# Cython formatter demo — exercises every supported construct.
# Python portions below have deliberate formatting issues for Black to correct.
# Cython declarations are preserved exactly; Python expressions are normalised.

# --- imports ---
from libc.math cimport sin, cos, sqrt
from libc.string cimport memcpy, memset
cimport numpy as cnp
import numpy as np

# --- type aliases ---
ctypedef double float64
ctypedef int int32

# --- C header wrapper ---
cdef extern from "math.h":
    double M_PI
    double M_E
    double pow(double base, double exp)

# --- struct / union / enum ---
cdef struct Point:
    double x
    double y

ctypedef struct Rect:
    double xmin
    double xmax
    double ymin
    double ymax

cdef union Data:
    int as_int
    double as_double

cdef enum Direction:
    NORTH
    SOUTH
    EAST
    WEST

# --- compound cdef block ---
cdef:
    int MAX_ITER = 1000
    double TOLERANCE = 1e-9
    int CHUNK_SIZE

# --- cdef class with cdef / cpdef methods ---
cdef class Vector2D:
    cdef double x, y

    def __init__( self,double x,double y ):
        self.x=x
        self.y=y

    cdef double norm(self):
        return sqrt(self.x**2+self.y**2)

    cpdef double dot( self,Vector2D other ) nogil:
        return self.x*other.x+self.y*other.y

    cpdef Vector2D scale( self,double factor ) with gil:
        return Vector2D(self.x*factor,self.y*factor)

# --- cdef / cpdef module-level functions ---
cdef double clamp( double val,double lo,double hi ) nogil:
    if val<lo:
        return lo
    if val>hi:
        return hi
    return val

cdef int argmax( double[:] arr,int n ) nogil:
    cdef int i, best = 0
    cdef double best_val = arr[0]
    for i in range(1,n):
        if arr[i]>best_val:
            best_val=arr[i]
            best=i
    return best

cpdef double trapz( double[:] y,double[:] x ):
    cdef int i, n = len(y)
    cdef double total = 0.0
    for i in range(1,n):
        total+=0.5*(y[i-1]+y[i])*(x[i]-x[i-1])
    return total

# --- plain def with typed args and memoryview ---
def normalise( double[:] arr,int n,bint inplace=True ):
    cdef int i
    cdef double lo, hi, rng
    lo=arr[0]; hi=arr[0]
    for i in range(1,n):
        if arr[i]<lo: lo=arr[i]
        if arr[i]>hi: hi=arr[i]
    rng=hi-lo
    if rng==0.0:
        return
    if inplace:
        for i in range(n):
            arr[i]=(arr[i]-lo)/rng
    else:
        result=np.empty(n)
        for i in range(n):
            result[i]=(arr[i]-lo)/rng
        return result

# --- 2-D memoryview ---
def mat_vec( double[:,:] A,double[:] x,double[:] out,int m,int n ):
    cdef int i, j
    for i in range(m):
        out[i]=0.0
        for j in range(n):
            out[i]+=A[i,j]*x[j]
