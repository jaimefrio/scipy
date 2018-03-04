#include "ni_support.h"
#include "ni_lbiter.h"


typedef void (extend_line_fn)(npy_double*, npy_intp, npy_intp, npy_intp,
                              npy_double);


struct LineBufferIterator_Internal {
    /* Extensions to input buffer before and after the line's content. */
    npy_intp before_len;
    npy_intp line_len;
    npy_intp after_len;
    /* Iterator over input and output arrays. */
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    char **dataptrs;
    /* Cached input and output array metadata. */
    npy_intp in_stride;
    int in_elsize;
    int in_is_swapped;
    PyArray_CopySwapNFunc *in_copyswapn_fn;
    PyArray_VectorUnaryFunc *in_cast_fn;
    npy_intp out_stride;
    int out_elsize;
    int out_is_swapped;
    PyArray_CopySwapNFunc *out_copyswapn_fn;
    PyArray_VectorUnaryFunc *out_cast_fn;
    /* Input and output buffers. */
    int reuse_buffer;
    npy_double *in_buffer;
    npy_double *in_buffer_start;
    char *in_buffer_write;
    npy_double *out_buffer;
    /* Extend mode. */
    extend_line_fn *extend_fn;
    npy_double extend_value;

};


static int
_LBIter_InitFilterSizes(LBIter *lbiter, npy_intp size, npy_intp origin)
{
    if (size < 1) {
        PyErr_SetString(PyExc_ValueError, "filter cannot be zero-sized");
        return NPY_FAIL;
    }

    lbiter->before_len = size / 2 + origin;
    lbiter->after_len = size - lbiter->before_len - 1;
    if (lbiter->before_len < 0 || lbiter->after_len < 0) {
        PyErr_Format(PyExc_ValueError,
                     "origin %zd out of bounds for filter of size %zd",
                     origin, size);
        return NPY_FAIL;
    }

    return NPY_SUCCEED;
}


static int
_LBIter_InitIterator(LBIter *lbiter, PyArrayObject *input, int axis,
                     PyArrayObject *output)
{
    PyArrayObject *ops[2] = {input, output};
    npy_uint32 op_flags[2] = {NPY_ITER_READONLY, NPY_ITER_WRITEONLY};

    lbiter->iter = NpyIter_MultiNew(2, ops, NPY_ITER_MULTI_INDEX,
                                    NPY_KEEPORDER, NPY_NO_CASTING,
                                    op_flags, NULL);

    if (lbiter->iter == NULL || !NpyIter_RemoveAxis(lbiter->iter, axis) ||
                                !NpyIter_RemoveMultiIndex(lbiter->iter)) {
        NpyIter_Deallocate(lbiter->iter);
        return NPY_FAIL;
    }

    lbiter->iternext = NpyIter_GetIterNext(lbiter->iter, NULL);
    if (lbiter->iternext == NULL) {
        NpyIter_Deallocate(lbiter->iter);
        return NPY_FAIL;
    }

    lbiter->dataptrs = NpyIter_GetDataPtrArray(lbiter->iter);
    return NPY_SUCCEED;
}


static int
_LBIter_CacheArrayMetadata(LBIter *lbiter, int axis)
{
    PyArrayObject **ops = NpyIter_GetOperandArray(lbiter->iter);
    PyArrayObject *input = ops[0];
    PyArrayObject *output = ops[1];
    PyArray_Descr *in_dtype = PyArray_DTYPE(input);
    PyArray_Descr *out_dtype = PyArray_DTYPE(output);
    PyArray_Descr *buffer_dtype = PyArray_DescrFromType(NPY_DOUBLE);

    lbiter->in_stride = PyArray_STRIDE(input, axis);
    lbiter->out_stride = PyArray_STRIDE(output, axis);
    lbiter->line_len = PyArray_DIM(input, axis);

    lbiter->in_elsize = in_dtype->elsize;
    lbiter->out_elsize = out_dtype->elsize;
    lbiter->in_is_swapped = PyArray_ISBYTESWAPPED(input);
    lbiter->out_is_swapped = PyArray_ISBYTESWAPPED(output);
    lbiter->in_copyswapn_fn = in_dtype->f->copyswapn;
    lbiter->out_copyswapn_fn = out_dtype->f->copyswapn;
    lbiter->in_cast_fn = in_dtype->f->cast[NPY_DOUBLE];
    lbiter->out_cast_fn = buffer_dtype->f->cast[out_dtype->type_num];

    Py_DECREF(buffer_dtype);
    return NPY_SUCCEED;
}


static int
_LBIter_InitBuffers(LBIter *lbiter)
{
    npy_intp extended_size = lbiter->before_len + lbiter->line_len +
                             lbiter->after_len;

    lbiter->in_buffer = malloc(sizeof(npy_double) * extended_size);
    if (lbiter->in_buffer == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }
    if (lbiter->reuse_buffer) {
        lbiter->out_buffer = lbiter->in_buffer + lbiter->before_len;
    }
    else {
        lbiter->out_buffer = malloc(sizeof(npy_double) * lbiter->line_len);
        if (lbiter->out_buffer == NULL) {
            PyErr_NoMemory();
            return NPY_FAIL;
        }
    }

    /*
     * The input buffer has three distinct areas: the line data area and
     * the two extension areas, before and after it. The in_buffer_start
     * pointers holds the address of the start of the line data area.
     *
     *   +-------------+----------------+------------+
     *   | before area | line data area | after area |
     *   +-------------+----------------+------------+
     *    ^             ^
     *    |             |
     *  in_buffer     in_buffer_start
     */
    lbiter->in_buffer_start = lbiter->in_buffer + lbiter->before_len;
    /*
     * To cast the input line to npy_double using the array's dtype cast
     * function, we first need to copy the data into a contiguous and
     * aligned memory segment. Rather than having yet another buffer for
     * this, we reuse the input buffer: in_buffer_write holds an address
     * such that copying a full uncasted input line starting there will
     * fill the line data area of the input buffer. Because npy_double
     * is larger than all supported types, and they all have power of
     * two sizes, in_buffer_write will be aligned to the input type's
     * size, and we can cast from in_buffer_write to in_buffer_start
     * without uncasted data being overwritten by the casted one.
     *
     *   +------------------------------------------------+---
     *   |                 line data area                 | after area
     *   +------------------------------------------------+---
     *    ^               ^                                ^
     *    |               |<-- len * sizeof(input_type) -->|
     *    |               |                                |
     *    |             in_buffer_write                    |
     *    |                                                |
     *    |<---------- len * sizeof(buffer_type) --------->|
     *    |
     *  in_buffer_start
     */
    lbiter->in_buffer_write =
            (char *)(lbiter->in_buffer_start + lbiter->line_len) -
            lbiter->in_elsize * lbiter->line_len;

    return NPY_SUCCEED;
}


static void
_extend_line_nearest(npy_double *buffer, npy_intp len, npy_intp before,
                     npy_intp after, npy_double unused_value)
{
    /* aaaaaaaa|abcd|dddddddd */
    npy_double * const first = buffer + before;
    npy_double * const last = first + len;
    npy_double *dst, val;

    val = *first;
    dst = buffer;
    while (before--) {
        *dst++ = val;
    }
    dst = last;
    val = *(last - 1);
    while (after--) {
        *dst++ = val;
    }
}


static void
_extend_line_wrap(npy_double *buffer, npy_intp len, npy_intp before,
                  npy_intp after, npy_double unused_value)
{
    /* abcdabcd|abcd|abcdabcd */
    npy_double * const first = buffer + before;
    npy_double * const last = first + len;
    const npy_double *src;
    npy_double *dst;

    src = last - 1;
    dst = first - 1;
    while (before--) {
        *dst-- = *src--;
    }
    src = first;
    dst = last;
    while (after--) {
        *dst++ = *src++;
    }
}


static void
_extend_line_reflect(npy_double *buffer, npy_intp len, npy_intp before,
                     npy_intp after, npy_double unused_value)
{
    /* abcddcba|abcd|dcbaabcd */
    npy_double * const first = buffer + before;
    npy_double * const last = first + len;
    const npy_double *src;
    npy_double *dst;

    src = first;
    dst = first - 1;
    while (before && src < last) {
        *dst-- = *src++;
        --before;
    }
    src = last - 1;
    while (before--) {
        *dst-- = *src--;
    }

    src = last - 1;
    dst = last;
    while (after && src >= first) {
        *dst++ = *src--;
        --after;
    }
    src = first;
    while (after--) {
        *dst++ = *src++;
    }
}


static void
_extend_line_mirror(npy_double *buffer, npy_intp len, npy_intp before,
                    npy_intp after, npy_double unused_value)
{
    /* abcddcba|abcd|dcbaabcd */
    npy_double * const first = buffer + before;
    npy_double * const last = first + len;
    const npy_double *src;
    npy_double *dst;

    src = first + 1;
    dst = first - 1;
    while (before && src < last) {
        *dst-- = *src++;
        --before;
    }
    src = last - 2;
    while (before--) {
        *dst-- = *src--;
    }
    src = last - 2;
    dst = last;
    while (after && src >= first) {
        *dst++ = *src--;
        --after;
    }
    src = first + 1;
    while (after--) {
        *dst++ = *src++;
    }
}


static void
_extend_line_constant(npy_double *buffer, npy_intp len, npy_intp before,
                      npy_intp after, npy_double value)
{
    /* abcddcba|abcd|dcbaabcd */
    npy_double * const last = buffer + before + len;
    npy_double *dst;

    dst = buffer;
    while (before--) {
        *dst++ = value;
    }    dst = last;
    while (after--) {
        *dst++ = value;
    }
}


static int
_LBIter_InitExtendMode(LBIter *lbiter, NI_ExtendMode mode, npy_double value)
{
    switch (mode) {
        case NI_EXTEND_NEAREST:
            lbiter->extend_fn = &_extend_line_nearest;
            break;
        case NI_EXTEND_WRAP:
            lbiter->extend_fn = &_extend_line_wrap;
            break;
        case NI_EXTEND_REFLECT:
            lbiter->extend_fn = &_extend_line_reflect;
            break;
        case NI_EXTEND_MIRROR:
            if (lbiter->line_len < 2) {
                /* 'mirror' is undefined for a single item array. */
                lbiter->extend_fn = &_extend_line_reflect;
            }
            else {
                lbiter->extend_fn = &_extend_line_mirror;
            }
            break;
        case NI_EXTEND_CONSTANT:
            lbiter->extend_fn = &_extend_line_constant;
            break;
        default:
            PyErr_Format(PyExc_ValueError, "unknown extend_mode %d", mode);
            return NPY_FAIL;
    }
    lbiter->extend_value = value;

    return NPY_SUCCEED;
}


static void
_LBIter_ReadLine(LBIter *lbiter)
{
    /*
     * TODO: If the input array line's are aligned, not swapped and
     * contiguous, we can skip the copyswap and cast directly into the
     * buffer. If the input array is of npy_double type, we can skip the
     * casting. If the extension mode is NI_EXTEND_CONSTANT, we only
     * need to extend once.
     */
    lbiter->in_copyswapn_fn(lbiter->in_buffer_write, lbiter->in_elsize,
                            lbiter->dataptrs[0], lbiter->in_stride,
                            lbiter->line_len, lbiter->in_is_swapped, NULL);
    lbiter->in_cast_fn(lbiter->in_buffer_write, lbiter->in_buffer_start,
                       lbiter->line_len, NULL, NULL);
    lbiter->extend_fn(lbiter->in_buffer, lbiter->line_len, lbiter->before_len,
                      lbiter->after_len, lbiter->extend_value);
}


static void
_LBIter_Write_Line(LBIter *lbiter)
{
    /* TODO: If the output array is of npy_double type, we can skip the
     * casting. If the output array line's are aligned, not swapped and
     * contiguous, we can skip the copyswap and cast directly into the
     * output array.
     */
    lbiter->out_cast_fn(lbiter->out_buffer, lbiter->out_buffer,
                        lbiter->line_len, NULL, NULL);
    lbiter->out_copyswapn_fn(lbiter->dataptrs[1], lbiter->out_stride,
                             lbiter->out_buffer, lbiter->out_elsize,
                             lbiter->line_len, lbiter->out_is_swapped, NULL);
}


LBIter*
LBIter_New(PyArrayObject *input, int axis, PyArrayObject *output,
           npy_intp filter_size, npy_intp filter_origin,
           NI_ExtendMode extend_mode, npy_double extend_value,
           int reuse_buffer)
{
    LBIter *lbiter = malloc(sizeof(LBIter));

    if (lbiter == NULL) {
        return (LBIter *)PyErr_NoMemory();
    }
    lbiter->iter = NULL;
    lbiter->in_buffer = NULL;
    lbiter->out_buffer = NULL;
    lbiter->reuse_buffer = reuse_buffer;

    if (!_LBIter_InitFilterSizes(lbiter, filter_size, filter_origin) ||
            !_LBIter_InitIterator(lbiter, input, axis, output) ||
            !_LBIter_CacheArrayMetadata(lbiter, axis) ||
            !_LBIter_InitBuffers(lbiter) ||
            !_LBIter_InitExtendMode(lbiter, extend_mode, extend_value)) {
        return LBIter_Delete(lbiter);
    }

    _LBIter_ReadLine(lbiter);


    return lbiter;
}


const npy_double*
LBIter_GetInputBuffer(LBIter *lbiter)
{
    return (const npy_double *)lbiter->in_buffer;
}


npy_double*
LBIter_GetOutputBuffer(LBIter *lbiter)
{
    return lbiter->out_buffer;
}


int
LBIter_Next(LBIter *lbiter)
{
    int iteration_not_finished;

    _LBIter_Write_Line(lbiter);
    iteration_not_finished = lbiter->iternext(lbiter->iter);
    if (iteration_not_finished) {
        _LBIter_ReadLine(lbiter);
    }

    return iteration_not_finished;
}


LBIter*
LBIter_Delete(LBIter *lbiter)
{
    NpyIter_Deallocate(lbiter->iter);
    free(lbiter->in_buffer);
    if (!lbiter->reuse_buffer) {
        free(lbiter->out_buffer);
    }
    free(lbiter);

    return NULL;
}
