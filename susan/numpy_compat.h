/* Compatibility shims: Cython 3.x generates NumPy-2.0 accessor calls;
 * define them as struct-field macros on NumPy < 2.0.                   */
#ifndef SUSAN_NUMPY_COMPAT_H
#define SUSAN_NUMPY_COMPAT_H

#include <numpy/numpyconfig.h>

#if NPY_MAJOR_VERSION < 2

#define PyDataType_ELSIZE(d)           ((d)->elsize)
#define PyDataType_ALIGNMENT(d)        ((d)->alignment)
#define PyDataType_FLAGS(d)            ((d)->flags)
#define PyDataType_FIELDS(d)           ((d)->fields)
#define PyDataType_NAMES(d)            ((d)->names)
#define PyDataType_SUBARRAY(d)         ((d)->subarray)

#define PyArray_MultiIter_NUMITER(m)   ((m)->numiter)
#define PyArray_MultiIter_SIZE(m)      ((m)->size)
#define PyArray_MultiIter_INDEX(m)     ((m)->index)
#define PyArray_MultiIter_NDIM(m)      ((m)->nd)
#define PyArray_MultiIter_DIMS(m)      ((m)->dimensions)
#define PyArray_MultiIter_ITERS(m)     ((void **)((m)->iters))

#endif  /* NPY_MAJOR_VERSION < 2 */
#endif  /* SUSAN_NUMPY_COMPAT_H */
