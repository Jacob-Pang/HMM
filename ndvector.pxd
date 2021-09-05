import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport abs, exp, log, isinf, isnan, INFINITY, NAN

ctypedef double dtype

ctypedef fused ntype:
    int
    double
    long

ctypedef fused mtype:
    int
    double
    long

ctypedef struct ndvector_dtype:
    dtype* vector_ptr
    int* shape_ptr
    long* dweight_ptr
    long size
    int ndim

ctypedef struct ndvector_int:
    int* vector_ptr
    int* shape_ptr
    long* dweight_ptr
    long size
    int ndim

ctypedef fused ndvector_type:
    ndvector_dtype
    ndvector_int

ctypedef fused mdvector_type:
    ndvector_dtype
    ndvector_int

# ndvector construct operations ----------------------------------------------------------------------
cdef inline ndvector_dtype ndvector_dtypeconstruct(int* shape_ptr, int ndim):
    cdef ndvector_dtype ndvector_construct

    ndvector_construct.shape_ptr  = shape_ptr
    ndvector_construct.ndim = ndim
    ndvector_init_size(&ndvector_construct)

    ndvector_construct.vector_ptr = <dtype*>malloc(sizeof(dtype) * ndvector_construct.size)
    ndvector_construct.dweight_ptr = <long*>malloc(sizeof(long) * ndvector_construct.size)
    ndvector_init_dweight(&ndvector_construct)

    return ndvector_construct

cdef inline ndvector_int ndvector_intconstruct(int* shape_ptr, int ndim):
    cdef ndvector_int ndvector_construct
    
    ndvector_construct.shape_ptr  = shape_ptr
    ndvector_construct.ndim = ndim
    ndvector_init_size(&ndvector_construct)

    ndvector_construct.vector_ptr = <int*>malloc(sizeof(int) * ndvector_construct.size)
    ndvector_construct.dweight_ptr = <long*>malloc(sizeof(long) * ndvector_construct.size)
    ndvector_init_dweight(&ndvector_construct)

    return ndvector_construct

cdef inline ndvector_int ndvector_identityconstruct(int n):
    cdef Py_ssize_t index
    cdef int* shape_ptr = <int*>malloc(sizeof(int) * 2)

    shape_ptr[0] = n
    shape_ptr[1] = n

    cdef ndvector_int ndvector_construct = ndvector_intconstruct(&shape_ptr[0], 2)
    ndvector_identitycast(ndvector_construct.vector_ptr, n)
    
    return ndvector_construct

cdef inline void ndvector_identitycast(int* vector_ptr, int n):
    cdef Py_ssize_t index

    for index in range(n ** 2):
        vector_ptr[index] = (index % (n + 1) == 0)

cdef inline void ndvector_deconstruct(ndvector_type* ndvector_ptr):
    free(ndvector_ptr.vector_ptr)
    free(ndvector_ptr.shape_ptr)
    free(ndvector_ptr.dweight_ptr)


cdef inline void ndvector_init_size(ndvector_type* ndvector_ptr):
    ndvector_ptr.size = vector_reduce(&vproduct_int, ndvector_ptr.shape_ptr, ndvector_ptr.ndim)

cdef inline void ndvector_init_copy(ndvector_type* ndvector_ptr, ndvector_type* ndvector_copyptr):
    if &ndvector_ptr == &ndvector_copyptr:
        return
    
    vector_init_copy(ndvector_ptr.vector_ptr,  ndvector_copyptr.vector_ptr,  ndvector_copyptr.size)
    vector_init_copy(ndvector_ptr.shape_ptr,   ndvector_copyptr.shape_ptr,   ndvector_copyptr.ndim)
    vector_init_copy(ndvector_ptr.dweight_ptr, ndvector_copyptr.dweight_ptr, ndvector_copyptr.ndim)

    ndvector_ptr.size = ndvector_copyptr.size
    ndvector_ptr.ndim = ndvector_copyptr.ndim

cdef inline void ndvector_init_dweight(ndvector_type* ndvector_ptr):
    compute_dweight(ndvector_ptr.dweight_ptr, ndvector_ptr.shape_ptr, ndvector_ptr.ndim)

cdef inline void compute_dweight(long* dweight_ptr, int* shape_ptr, int ndim):
    cdef Py_ssize_t index

    dweight_ptr[ndim - 1] = 1

    for index in range(ndim - 1, 0, -1):
        dweight_ptr[index - 1] = (dweight_ptr[index] * shape_ptr[index])


cdef inline void vector_init_repeat(ntype* vector_ptr, ntype x, long size):
    cdef Py_ssize_t index

    for index in range(size):
        vector_ptr[index] = x

cdef inline void vector_init_zeros(ntype* vector_ptr, long size):
    vector_init_repeat(vector_ptr, 0, size)

cdef inline void vector_init_copy(ntype* vector_ptr, ntype* copyvector_ptr, long size):
    cdef Py_ssize_t index

    for index in range(size):
        vector_ptr[index] = copyvector_ptr[index]


cdef inline int vector_equiv(ntype* vector_xptr, ntype* vector_yptr, long size):
    cdef Py_ssize_t index

    for index in range(size):
        if vector_xptr[index] != vector_yptr[index]:
            return 0
    
    return 1

cdef inline ptr_to_nparray_dtype(dtype* ptr, long size):
    return np.asarray(<np.double_t[:size]> ptr)

# vmath operations -----------------------------------------------------------------------------------
cdef inline int isneginf(dtype x):
    return isinf(x) and x < 0

cdef inline int isposinf(dtype x):
    return isinf(x) and x > 0

cdef inline dtype vconstruct_dtype(dtype x, dtype y):
    return y

cdef inline dtype vmax_dtype(dtype x, dtype y):
    return max(x, y)

cdef inline int vmax_int(int x, int y):
    return max(x, y)

cdef inline long vmin_long(long x, long y):
    return min(x, y)

cdef inline dtype vsum_dtype(dtype x, dtype y):
    return x + y

cdef inline dtype vsubtract_dtype(dtype x, dtype y):
    return x - y

cdef inline dtype vrevsubtract_dtype(dtype x, dtype y):
    return y - x

cdef inline int vproduct_int(int x, int y):
    return x * y

cdef inline dtype vproduct_dtype_int(dtype x, int y):
    if x == 0 or y == 0:
        return 0

    return <dtype>(x * y)

cdef inline dtype vproduct_dtype(dtype x, dtype y):
    return x * y

cdef inline dtype vdivide_dtype(dtype x, dtype y):
    return x / y

cdef inline dtype logprodexp(dtype x, dtype y):
    if isneginf(x) or isneginf(y):
        return -INFINITY
    elif isposinf(x) or isposinf(y):
        return INFINITY
    
    return x + y

cdef inline dtype logdivexp(dtype x, dtype y):
    return logprodexp(x, -y)

# vector operations ----------------------------------------------------------------------------------
cdef inline ntype vector_reduce(ntype(*operation)(ntype, ntype), ntype* vector_ptr, long size):
    cdef ntype vreduced = vector_ptr[0]
    cdef Py_ssize_t index

    for index in range(1, size):
        vreduced = operation(vreduced, vector_ptr[index])
    
    return vreduced

cdef inline dtype vreduce_dtypemax(dtype* vector_ptr, long size):
    return vector_reduce(&vmax_dtype, vector_ptr, size)

cdef inline int vreduce_dtype_argmax(dtype* vector_ptr, long size):
    cdef dtype vmax = vreduce_dtypemax(vector_ptr, size)
    cdef Py_ssize_t index

    for index in range(size):
        if vector_ptr[index] == vmax:
            return index
    
    return -1

cdef inline dtype vreduce_dtypesum(dtype* vector_ptr, long size):
    return vector_reduce(&vsum_dtype, vector_ptr, size)

cdef inline dtype vreduce_logsumexp(dtype* vector_ptr, long size):
    cdef dtype vmax = vector_reduce(&vmax_dtype, vector_ptr, size)

    if isinf(vmax):
        return -INFINITY
    
    cdef dtype vsum = 0
    cdef Py_ssize_t index

    for index in range(size):
        vsum += exp(vector_ptr[index] - vmax)
    
    return log(vsum) + vmax

cdef inline void vector_broadcast(ntype(*operation)(ntype, ntype), ntype* vector_outptr, ntype* vector_xptr,
    long size):

    cdef Py_ssize_t index

    for index in range(size):
        vector_outptr[index] = operation(vector_outptr[index], vector_xptr[index])
        
# ndvector operations --------------------------------------------------------------------------------
cdef inline ntype ndvector_indexing_vptr(ntype* ndvector_xvptr, int* indexes, long* xdweight_ptr, int xndim):
    cdef Py_ssize_t index
    cdef long ndvector_index = 0

    for index in range(xndim):
        ndvector_index += xdweight_ptr[index] * indexes[index]
    
    return ndvector_xvptr[ndvector_index]
    
cdef inline void ndvector_reducecast_vptr(ntype(*vreduce_operation)(mtype*, long), ntype* ndvector_outvptr,
    mtype* ndvector_xvptr, int* xshape_ptr, long* xdweight_ptr, long xsize, int xndim, int axis):
    if axis < 0:
        axis = xndim - 1

    cdef Py_ssize_t outptr_index = 0, axis_index
    cdef long axis_size = xshape_ptr[axis]
    cdef long outptr_size = xsize // axis_size

    if axis == (xndim - 1):
        for outptr_index in range(outptr_size):
            ndvector_outvptr[outptr_index] = vreduce_operation(&ndvector_xvptr[outptr_index
                    * axis_size], axis_size)
            
        return

    cdef Py_ssize_t x
    cdef long axis_stride = xdweight_ptr[axis]
    cdef int* statevector_ptr = <int*>malloc(sizeof(int) * xsize)
    cdef mtype* axisvector_ptr = <mtype*>malloc(sizeof(mtype) * axis_size)

    vector_init_zeros(statevector_ptr, xsize)
    outptr_index = 0

    for x in range(xsize):
        if statevector_ptr[x]:
            continue
        
        for axis_index in range(axis_size):
            axisvector_ptr[axis_index] = ndvector_xvptr[x + axis_index * axis_stride]
            statevector_ptr[x + axis_index * axis_stride] = 1

        ndvector_outvptr[outptr_index] = vreduce_operation(axisvector_ptr, axis_size)
        outptr_index += 1

    free(statevector_ptr)
    free(axisvector_ptr)

cdef inline void ndvector_mreducecast_vptr(ntype(*vreduce_operation)(ntype*, long), ntype* ndvector_outvptr,
    ntype* ndvector_xvptr, int* xshape_ptr, long* xdweight_ptr, long xsize, int xndim, int order,
    int axis):

    if order == xndim:
        vector_init_copy(ndvector_outvptr, ndvector_xvptr, xsize)
        return

    cdef int reshape_ndim = order + 1
    cdef int* reshape_ptr = <int*>malloc(sizeof(int) * reshape_ndim)
    cdef long* reshape_dweight_ptr = <long*>malloc(sizeof(long) * reshape_ndim)

    if axis == 0:
        reshape_ptr[0] = vector_reduce(&vproduct_int, xshape_ptr, xndim - reshape_ndim + 1)
        vector_init_copy(&reshape_ptr[1], &xshape_ptr[xndim - reshape_ndim + 1], reshape_ndim - 1)
    else:
        axis = -1
        vector_init_copy(reshape_ptr, xshape_ptr, reshape_ndim - 1)
        reshape_ptr[reshape_ndim - 1] = vector_reduce(&vproduct_int, &reshape_ptr[reshape_ndim - 1],
                xndim - reshape_ndim + 1)
    
    compute_dweight(reshape_dweight_ptr, reshape_ptr, reshape_ndim)
    ndvector_reducecast_vptr(vreduce_operation, ndvector_outvptr, ndvector_xvptr, reshape_ptr,
            reshape_dweight_ptr, xsize, reshape_ndim, axis)
    
    free(reshape_ptr)
    free(reshape_dweight_ptr)

cdef inline void ndvector_broadcast_vptr(ntype(*operation)(ntype, mtype), ntype* ndvector_vptr,
    mtype* mdvector_vector_ptr, long ndvector_size, long mdvector_size, int axis):

    if ntype is mtype and &ndvector_vptr == &mdvector_vector_ptr:
        return

    assert ndvector_size % mdvector_size == 0
    cdef long broadcast_x  = ndvector_size // mdvector_size
    cdef Py_ssize_t x, y, index

    if axis == 0:
        for x in range(broadcast_x):
            for y in range(mdvector_size):
                index = x * mdvector_size + y
                ndvector_vptr[index] = operation(ndvector_vptr[index], mdvector_vector_ptr[y])
    else:
        for y in range(mdvector_size):
            for x in range(broadcast_x):
                index = y * broadcast_x + x
                ndvector_vptr[index] = operation(ndvector_vptr[index], mdvector_vector_ptr[y])

# dtype ndvector operations --------------------------------------------------------------------------
cdef inline void ndvector_dtype_reducecast(dtype(*vreduce_operation)(dtype*, long),
    ndvector_dtype* ndvector_outptr, ndvector_dtype* ndvector_xptr, int axis):
    
    if axis < 0:
        axis = ndvector_xptr.ndim - 1

    cdef Py_ssize_t outptr_index = 0, axis_index

    ndvector_outptr.ndim = ndvector_xptr.ndim - 1
    ndvector_outptr.size = ndvector_xptr.size // ndvector_xptr.shape_ptr[axis]

    for axis_index in range(ndvector_xptr.ndim):
        if axis_index != axis:
            ndvector_outptr.shape_ptr[outptr_index] = ndvector_xptr.shape_ptr[axis_index]
            outptr_index += 1
    
    ndvector_init_dweight(ndvector_outptr)
    ndvector_reducecast_vptr(vreduce_operation, ndvector_outptr.vector_ptr, ndvector_xptr.vector_ptr,
            ndvector_xptr.shape_ptr, ndvector_xptr.dweight_ptr, ndvector_xptr.size,
            ndvector_xptr.ndim, axis)

cdef inline void ndvector_dtype_mreducecast(dtype(*vreduce_operation)(dtype*, long),
    ndvector_dtype* ndvector_outptr, ndvector_dtype* ndvector_xptr, int order, int axis):

    ndvector_outptr.ndim = order
    
    if axis == 0:
        vector_init_copy(ndvector_outptr.shape_ptr, &ndvector_xptr.shape_ptr
                [ndvector_xptr.ndim - order], order)
    else:
        vector_init_copy(ndvector_outptr.shape_ptr, ndvector_xptr.shape_ptr, order)

    ndvector_init_size(ndvector_outptr)
    ndvector_init_dweight(ndvector_outptr)
    ndvector_mreducecast_vptr(vreduce_operation, ndvector_outptr.vector_ptr, ndvector_xptr.vector_ptr,
            ndvector_xptr.shape_ptr, ndvector_xptr.dweight_ptr, ndvector_xptr.size,
            ndvector_xptr.ndim, order, axis)


cdef inline void ndvector_dtype_identitycast(ndvector_dtype* ndvector_outptr, ndvector_dtype* ndvector_xptr, int order):
    cdef int n = vector_reduce(&vmax_int, ndvector_xptr.shape_ptr, ndvector_xptr.size)

    ndvector_dtype_identitycast_vptr(ndvector_outptr.vector_ptr, ndvector_xptr.vector_ptr, ndvector_xptr.size,
            ndvector_xptr.ndim, n, order)

    ndvector_outptr.ndim = order
    ndvector_outptr.size = n ** order

    vector_init_repeat(ndvector_outptr.shape_ptr, n, order)
    ndvector_init_dweight(ndvector_outptr)

cdef inline void ndvector_dtype_identitycast_vptr(dtype* ndvector_outvptr, dtype* ndvector_xvptr,
    long xsize, int xndim, int n, int order):

    if xndim == order:
        vector_init_copy(ndvector_outvptr, ndvector_xvptr, xsize)
        return
    
    cdef Py_ssize_t mdim
    cdef long identvector_size = n ** 2, mdvector_size
    cdef dtype* mdvector_identity
    cdef int* identvector_ptr = <int*>malloc(sizeof(int) * identvector_size)

    ndvector_identitycast(identvector_ptr, n)
    vector_init_copy(ndvector_outvptr, ndvector_xvptr, xsize)

    for mdim in range(xndim + 1, order + 1):
        mdvector_size = n ** mdim
        mdvector_identity = <dtype*>malloc(sizeof(dtype) * mdvector_size)

        ndvector_broadcast_vptr(&vconstruct_dtype, mdvector_identity, ndvector_outvptr,
                mdvector_size, xsize, -1)
            
        ndvector_broadcast_vptr(&vproduct_dtype_int, mdvector_identity, identvector_ptr,
                mdvector_size, identvector_size, 0)
        
        vector_init_copy(ndvector_outvptr, mdvector_identity, mdvector_size)

        free(mdvector_identity)
        xsize = mdvector_size

    free(identvector_ptr)

cdef inline void ndvector_logdtype_identitycast_vptr(dtype* ndvector_outvptr, dtype* ndvector_xvptr,
    long xsize, int xndim, int n, int order):

    cdef Py_ssize_t index
    cdef dtype* ndvector_yvptr = <dtype*>malloc(sizeof(dtype) * xsize)

    for index in range(xsize):
        if ndvector_xvptr[index] == 0:
            ndvector_yvptr[index] = NAN
        else:
            ndvector_yvptr[index] = ndvector_xvptr[index]
    
    ndvector_dtype_identitycast_vptr(ndvector_outvptr, ndvector_yvptr, xsize, xndim, n, order)

    for index in range(n ** order):
        if ndvector_outvptr[index] == 0:
            ndvector_outvptr[index] = -INFINITY
        elif isnan(ndvector_outvptr[index]):
            ndvector_outvptr[index] = 0

    free(ndvector_yvptr)

cdef inline void ndvector_mdargmax(int* argmax_outptr, dtype* ndvector_xvptr, int* xshape_ptr, long* xdweight_ptr,
    long xsize, int xndim):

    if xndim == 1:
        argmax_outptr[0] = vreduce_dtype_argmax(ndvector_xvptr, xsize)
        return

    cdef long mdvector_size = xsize // xshape_ptr[xndim - 1]
    cdef int* argmax_tempbuffer = <int*>malloc(sizeof(int) * mdvector_size)
    cdef dtype* mdvector_vector_ptr = <dtype*>malloc(sizeof(dtype) * mdvector_size)
    cdef long* mdvector_dweight_ptr = <long*>malloc(sizeof(long) * (xndim - 1))

    ndvector_reducecast_vptr(&vreduce_dtype_argmax, argmax_tempbuffer, ndvector_xvptr, xshape_ptr,
            xdweight_ptr, xsize, xndim, -1)
    ndvector_reducecast_vptr(&vreduce_dtypemax, mdvector_vector_ptr, ndvector_xvptr, xshape_ptr,
            xdweight_ptr, xsize, xndim, -1)
    
    compute_dweight(mdvector_dweight_ptr, xshape_ptr, xndim - 1)
    ndvector_mdargmax(argmax_outptr, mdvector_vector_ptr, xshape_ptr, mdvector_dweight_ptr,
            mdvector_size, xndim - 1)

    argmax_outptr[xndim - 1] = ndvector_indexing_vptr(argmax_tempbuffer, argmax_outptr,
            mdvector_dweight_ptr, xndim - 1)

    free(argmax_tempbuffer)
    free(mdvector_vector_ptr)
    free(mdvector_dweight_ptr)
