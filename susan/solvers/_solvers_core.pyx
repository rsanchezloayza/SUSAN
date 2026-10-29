# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
from libc.math cimport fabs, fmax
from libc.stdlib cimport malloc, free

cdef double CHI2_2_MEDIAN = 1.3863


cdef double _wmedian(double[::1] v, double[::1] w, Py_ssize_t[::1] order):
    """weighted median (first value whose cumulative weight reaches half), as np.searchsorted on the cumsum"""
    cdef Py_ssize_t n = v.shape[0], i
    cdef double tot = 0, c = 0, half
    for i in range(n):
        tot += w[i]
    if n == 0 or tot <= 0:
        return 1.0
    o = np.argsort(np.asarray(v), kind='quicksort')
    for i in range(n):
        order[i] = o[i]
    half = 0.5*tot
    for i in range(n):
        c += w[order[i]]
        if c >= half:
            return v[order[i]]
    return v[order[n-1]]


cdef int _solve(double*A, double*b, int m) noexcept nogil:
    """A x = b in place (b <- x), LU with partial pivoting; A (m x m) row major is overwritten."""
    cdef int i, j, k, p
    cdef double mx, t, f
    for k in range(m):
        p = k
        mx = fabs(A[k*m+k])
        for i in range(k+1,m):
            if fabs(A[i*m+k]) > mx:
                mx = fabs(A[i*m+k])
                p = i
        if mx == 0:
            return -1
        if p != k:
            for j in range(m):
                t = A[k*m+j]; A[k*m+j] = A[p*m+j]; A[p*m+j] = t
            t = b[k]; b[k] = b[p]; b[p] = t
        for i in range(k+1,m):
            f = A[i*m+k]/A[k*m+k]
            if f != 0:
                for j in range(k,m):
                    A[i*m+j] -= f*A[k*m+j]
                b[i] -= f*b[k]
    for i in range(m-1,-1,-1):
        t = b[i]
        for j in range(i+1,m):
            t -= A[i*m+j]*b[j]
        b[i] = t/A[i*m+i]
    return 0


def fit_robust(double[:, :, ::1] D, double[:, :, ::1] I, double[:, ::1] O, double[::1] P, double[::1] th0,
               int iters, double ridge, bint frozen):
    """Robust (Cauchy, scale 2 sigma) weighted least squares of a correction field from th0.

    D (n, 2, nP) design, I (n, 2, 2) precisions, O (n, 2) observations, P (n,) prior weights. The scale kappa is the
    P-weighted median of the squared Mahalanobis residuals / the chi^2_2 median. A ridge on the grid nodes (parameters
    3..) scaled by their mean diagonal. frozen: the rigid part (first 3 parameters) stays at th0, only the nodes are
    fitted. Returns (th, kappa)."""
    cdef Py_ssize_t n = D.shape[0], nP = D.shape[2]
    cdef int p0 = 3 if frozen else 0
    cdef int m = nP - p0
    cdef int it, i, q, jq, jp, pp
    cdef double r0, r1, kap = 1.0, t0, t1, d0, d1, M0, M1, reg, tr
    th_np = np.array(th0, dtype=np.float64)
    cdef double[::1] th = th_np
    m2_np = np.empty(n)
    cdef double[::1] m2 = m2_np
    cdef double[::1] w = np.empty(n)
    cdef Py_ssize_t[::1] order = np.empty(n, dtype=np.intp)
    cdef long[:, ::1] nz = np.empty((n, max(m,1)), dtype=np.int64)
    cdef long[::1] cnt = np.zeros(n, dtype=np.int64)
    cdef double*A = <double*>malloc(sizeof(double)*max(m,1)*max(m,1))
    cdef double*b = <double*>malloc(sizeof(double)*max(m,1))
    try:
        for i in range(n):
            for q in range(m):
                if D[i,0,p0+q] != 0 or D[i,1,p0+q] != 0:
                    nz[i,cnt[i]] = q
                    cnt[i] += 1
        for it in range(iters+1):
            for i in range(n):
                r0 = O[i,0]
                r1 = O[i,1]
                for q in range(nP):
                    r0 -= D[i,0,q]*th[q]
                    r1 -= D[i,1,q]*th[q]
                m2[i] = r0*(I[i,0,0]*r0 + I[i,0,1]*r1) + r1*(I[i,1,0]*r0 + I[i,1,1]*r1)
            kap = fmax(_wmedian(m2, P, order)/CHI2_2_MEDIAN, 1e-12)
            if it == iters:
                return th_np, kap
            for i in range(n):
                w[i] = P[i]/(1 + m2[i]/(4*kap))
            for q in range(m*m):
                A[q] = 0
            for q in range(m):
                b[q] = 0
            for i in range(n):
                if w[i] == 0:
                    continue
                t0 = O[i,0]
                t1 = O[i,1]
                if frozen:
                    for q in range(3):
                        t0 -= D[i,0,q]*th[q]
                        t1 -= D[i,1,q]*th[q]
                for jq in range(cnt[i]):
                    q = nz[i,jq]
                    d0 = D[i,0,p0+q]
                    d1 = D[i,1,p0+q]
                    M0 = w[i]*(I[i,0,0]*d0 + I[i,0,1]*d1)
                    M1 = w[i]*(I[i,1,0]*d0 + I[i,1,1]*d1)
                    b[q] += M0*t0 + M1*t1
                    for jp in range(cnt[i]):
                        pp = nz[i,jp]
                        A[pp*m+q] += D[i,0,p0+pp]*M0 + D[i,1,p0+pp]*M1
            if frozen:
                tr = 0
                for q in range(m):
                    tr += A[q*m+q]
                reg = ridge*tr/m + 1e-12
                for q in range(m):
                    A[q*m+q] += reg
            elif nP > 3:
                tr = 0
                for q in range(3,nP):
                    tr += A[q*m+q]
                reg = ridge*tr/(nP-3) + 1e-12
                for q in range(3,nP):
                    A[q*m+q] += reg
            for q in range(m):
                A[q*m+q] += 1e-12
            if _solve(A,b,m) != 0:
                raise np.linalg.LinAlgError('singular robust fit')
            for q in range(m):
                th[p0+q] = b[q]
        return th_np, 1.0
    finally:
        free(A)
        free(b)
