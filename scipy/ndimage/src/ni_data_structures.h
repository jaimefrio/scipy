#ifndef NI_DATA_STRUCTURES_H
#define NI_DATA_STRUCTURES_H


typedef struct CoordinateStack_Internal CoordinateStack;

/* Creates a new, empty CoordinateStack instance. */
CoordinateStack*
CoS_New(int ndim);

/* Returns 1 if the stacks is empty, 0 if not. */
int
CoS_IsEmpty(CoordinateStack *costack);

/* Pushes coordinates onto the stack. Returns 0 on success. */
int
CoS_PushCoords(CoordinateStack *costack, const npy_intp *coords);

/* Pushes sum of coordinates onto the stack. Returns 0 on success. */
int
CoS_PushSumOfCoords(CoordinateStack *costack, const npy_intp *coords1,
                    const npy_intp *coords2);

/* Returns a pointer to the coordinates at the top of the stack. */
const npy_intp*
CoS_PeekCoords(CoordinateStack *costack);

/* Removes the coordinates at the top of the stack. Returns 0 on success. */
int
CoS_PopCoords(CoordinateStack *costack);

/* Releases all memory of a CoordinateStack instance. */
void
CoS_Delete(CoordinateStack *costack);

void
CoS_PrintDebugInfo(CoordinateStack *costack);


#endif  /*  NI_DATA_STRUCTURES_H */
