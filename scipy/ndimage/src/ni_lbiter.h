/*
 * LBIter is the L(ine) B(uffer) Iter(ator). It simplifies iterating
 * over lines of an input array along an axis, extended beyond the
 * boundaries according to an extension mode, and write a processed
 * result to the corresponding line of an output array.
 *
 * LBiter holds two buffers of npy_double type. When initialized with a
 * call to LBIter_New, the first line of the input array is read, casted
 * to npy_double, copied into the input buffer and extended beyond the
 * boundaries. Once this data has been processed and the results written
 * to the output buffer, a call to LBIter_Next finishes iteration over
 * this line: the data in the output buffer is casted to the output
 * array's type and copied to the appropriate line, and a new line from
 * the input array is read, casted and extended into the input buffer.
 *
 * The typical structure of code using LBIter is:
 *
 *   LBIter *lbiter;
 *   const double *in_buffer;
 *   double *out_buffer;
 *
 *   lbiter = LBIter_New(input, axis, output, filter_size, filter_origin,
 *                       extend_mode, extend_value, reuse_buffer);
 *   if (lbiter == NULL) {
 *      return NULL;
 *   }
 *
 *   in_buffer = LBIter_GetInputBuffer(lbiter);
 *   out_buffer = LBIter_GetOutputBuffer(lbiter);
 *
 *   do {
 *      // Your code reading from in_buffer and writing to out_buffer.
 *   } while (LBIter_Next(lbiter));
 *
 *   LBIter_Delete(lbiter);
 */


#ifndef NI_LBITER_H
#define NI_LBITER_H


/* The struct internals are a private implementation detail. */
typedef struct LineBufferIterator_Internal LBIter;


/*
 * Allocates a new line buffer iterator to iterate over the lines on
 * the given axis of input and output. The input buffer will be extended
 * as requested by the filter size and origin, and the extension mode
 * and value. If reuse buffer is non-zero, the input and output buffers
 * will share the same memory.
 *
 *  input - The input array.
 *  axis - The axis of the lines of input to iterate over.
 *  output - The output array, must be of the same shape as input.
 *  filter_size - The size of the filter. This will determine how much
 *      the input lines will be extended beyond their original size. If
 *      no extension is required, set it to zero.
 *  filter_origin - The offset of the filter from centered. This will
 *      determine how the extension of the input lines is split between
 *      before and after the actual data of the input line. Set it to
 *      zero for a balanced split.
 *  extend_mode - The method to use to extend the input lines beyond
 *      their boundaries, see NI_ExtendMode for details.
 *  extend_value - The value to use for extension if NI_EXTEND_CONSTANT
 *      is the extension mode.
 *  reuse_buffer - If non-zero, a single buffer will be allocated, and
 *      will be shared by the input and output buffers, so when writing
 *      to the output buffer the input data will be overwritten. Set it
 *      to zero to get two separate buffers.
 */
LBIter*
LBIter_New(PyArrayObject *input, int axis, PyArrayObject *output,
           npy_intp filter_size, npy_intp filter_origin,
           NI_ExtendMode extend_mode, npy_double extend_value,
           int reuse_buffer);


/*
 * Returns a pointer to the beginning of the input buffer.
 *
 * Notice that this is not the beginning of the data copied from the
 * input array line, but of the data extended before that:
 *
 *  +--------+----------------------+-------+
 *  | before | data from array line | after |
 *  +--------+----------------------+-------+
 *   ^
 *   |
 *   LBIter_GetInputBuffer
 */
const npy_double*
LBIter_GetInputBuffer(LBIter* lbiter);


/*
 * Returns a pointer to the beginning of the output buffer.
 */
npy_double*
LBIter_GetOutputBuffer(LBIter* lbiter);


/*
 * Copies the output buffer to the output array, tries to advance the
 * iterator to the next line and, if it succeeds, loads and extends a
 * new line from the input array into the input buffer. Returns NPY_FAIL
 * if the iterator is exhausted, NPY_SUCCEED if a new line is read.
 */
int
LBIter_Next(LBIter *lbiter);


/*
 * Releases all memory allocated by LBIter_New. Always returns NULL.
 */
LBIter*
LBIter_Delete(LBIter *lbiter);


#endif
