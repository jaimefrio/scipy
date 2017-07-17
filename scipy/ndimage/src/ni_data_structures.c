#include "ni_support.h"
#include "ni_data_structures.h"


#define INITIAL_STACK_SIZE 64

struct CoordinateStack_Internal {
    /* Array of stored coordinates. */
    npy_intp *stack;
    /* Allocated size of the stack. */
    npy_intp max_size;
    /* Number of used entries on the stack. */
    npy_intp size;
    /* Number of dimensions of coordinates stored on stack. */
    int ndim;
};

CoordinateStack*
CoS_New(int ndim) {
    CoordinateStack* costack;
    if (ndim < 1 || ndim >= NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                     "number of dimensions (%d) must be within 1 and %d",
                     ndim, NPY_MAXDIMS);
        return NULL;
    }
    costack = malloc(sizeof(CoordinateStack));
    if (costack == NULL) {
        return (CoordinateStack *)PyErr_NoMemory();
    }
    costack->ndim = ndim;
    costack->max_size = INITIAL_STACK_SIZE * costack->ndim;
    costack->stack = malloc(sizeof(npy_intp) * costack->max_size);
    if (costack->stack == NULL) {
        free(costack);
        return (CoordinateStack *)PyErr_NoMemory();
    }
    costack->size = 0;


    return costack;
}
int
CoS_IsEmpty(CoordinateStack *costack)
{
    return costack->size == 0;
}

int
CoS_PushCoords(CoordinateStack *costack, const npy_intp *coords)
{
    if (costack->size + costack->ndim > costack->max_size) {
        const npy_intp new_max_size = costack->max_size * 2;
        npy_intp *new_stack = realloc(costack->stack,
                                      sizeof(npy_intp) * new_max_size);
        if (new_stack == NULL) {
            PyErr_NoMemory();
            return 1;
        }
        costack->stack = new_stack;
        costack->max_size = new_max_size;
    }
    memcpy(costack->stack + costack->size, coords,
           sizeof(npy_intp) * costack->ndim);
    costack->size += costack->ndim;
    return 1;
}

int
CoS_PushSumOfCoords(CoordinateStack *costack,
                    const npy_intp *coords1, const npy_intp *coords2)
{
    npy_intp sum_of_coords[NPY_MAXDIMS];
    int dimension;

    for (dimension = 0; dimension < costack->ndim; ++dimension) {
        sum_of_coords[dimension] = *coords1++ + *coords2++;
    }

    return CoS_PushCoords(costack, sum_of_coords);
}

const npy_intp*
CoS_PeekCoords(CoordinateStack *costack)
{
    if (costack->size - costack->ndim < 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot peek into an empty CoordinateStack");
        return NULL;
    }
    return costack->stack + costack->size - costack->ndim;
}

int
CoS_PopCoords(CoordinateStack *costack)
{
    if (costack->size - costack->ndim < 0){
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot pop an empty CoordinateStack");
        return 1;
    }
    costack->size -= costack->ndim;
}

void
CoS_Delete(CoordinateStack *costack)
{
    if (costack != NULL) {
        free(costack->stack);
    }
    free(costack);
}

void
CoS_PrintDebugInfo(CoordinateStack *costack)
{
    npy_intp entries = costack->size;
    npy_intp j = 0;
    int dim;
    npy_intp *ptr = costack->stack;
    printf("CoordinateStack object at %p.\n", costack);
    printf("Stack at address %p.\n", costack->stack);
    printf("Stored coordinates are %dD.\n", costack->ndim);
    printf("Total stack capacity is %zd.\n", costack->max_size);
    printf("Of which %zd are in use:\n", costack->size);
    while (entries) {
        printf("#%zd: (", j++);
        for (dim = 0; dim < costack->ndim; ++dim) {
            printf("%zd%s", *ptr++, dim == costack->ndim - 1 ? "": ", ");
            entries--;
        }
        printf(")\n");
    }
    printf("\n");
}
