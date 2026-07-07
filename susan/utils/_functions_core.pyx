# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cos, sin, atan2, fabs, ceil, floor
from libc.stdlib cimport malloc, free


def radial_average(double[:, :, ::1] v):
    cdef int nz = v.shape[0], ny = v.shape[1], nx = v.shape[2]
    cdef int N = max(max(nz, ny), nx) // 2 + 1
    cdef int cnt_z = nz // 2, cnt_y = ny // 2, cnt_x = nx // 2
    cdef double[::1] val = np.zeros(N)
    cdef double[::1] wgt = np.zeros(N)
    cdef int k, j, i, r
    cdef double x, y, z, dr

    for k in range(nz):
        z = k - cnt_z
        for j in range(ny):
            y = j - cnt_y
            for i in range(nx):
                x = i - cnt_x
                dr = sqrt(x*x + y*y + z*z)
                r = <int>(dr + 0.5)
                if r < N:
                    val[r] += v[k, j, i]
                    wgt[r] += 1.0
    for r in range(N):
        if wgt[r] > 0:
            val[r] = val[r] / wgt[r]
    return np.asarray(val)


def radial_expansion(double[::1] arr):
    cdef int cnt = arr.shape[0] - 1
    cdef int N = 2 * cnt
    out = np.zeros((N, N, N), dtype=np.float32)
    cdef float[:, :, ::1] rslt = out
    cdef int k, j, i, r
    cdef double x, y, z, dr

    for k in range(N):
        z = k - cnt
        for j in range(N):
            y = j - cnt
            for i in range(N):
                x = i - cnt
                dr = sqrt(x*x + y*y + z*z)
                r = <int>(dr + 0.5)
                if r < cnt:
                    rslt[k, j, i] = <float>arr[r]
    return out


def _core_apply_fourier_rad_wgt(double complex[:, :, ::1] v_fou, float[::1] wgt):
    cdef double c_z = v_fou.shape[0] / 2.0
    cdef double c_y = v_fou.shape[1] / 2.0
    cdef int nz = v_fou.shape[0], ny = v_fou.shape[1], nx = v_fou.shape[2]
    cdef int wgt_max = wgt.shape[0] - 1
    cdef int z, y, x, r
    cdef double dz, dy, dx

    for z in range(nz):
        dz = (z - c_z) * (z - c_z)
        for y in range(ny):
            dy = (y - c_y) * (y - c_y)
            for x in range(nx):
                dx = <double>x * x
                r = <int>sqrt(dx + dy + dz)
                if r > wgt_max:
                    r = wgt_max
                v_fou[z, y, x] = wgt[r] * v_fou[z, y, x]


def _fsc_get_core(float[:, :, ::1] n, float[:, :, ::1] d1, float[:, :, ::1] d2):
    cdef int sz = n.shape[2]
    cdef float[::1] rslt = np.zeros(sz, dtype=np.float32)
    cdef float[::1] tmp1 = np.zeros(sz, dtype=np.float32)
    cdef float[::1] tmp2 = np.zeros(sz, dtype=np.float32)
    cdef int k, j, i, r
    cdef double z, y, dr, denom

    for k in range(n.shape[0]):
        z = k - n.shape[0] // 2
        for j in range(n.shape[1]):
            y = j - n.shape[1] // 2
            for i in range(n.shape[2]):
                dr = sqrt(<double>i*i + y*y + z*z)
                r = <int>(dr + 0.5)
                if r < sz:
                    rslt[r] += n[k, j, i]
                    tmp1[r] += d1[k, j, i]
                    tmp2[r] += d2[k, j, i]
    rslt[0] = 1.0
    for r in range(1, sz):
        denom = sqrt(<double>tmp1[r] * tmp2[r])
        if denom > 0:
            rslt[r] = <float>(rslt[r] / denom)
        else:
            rslt[r] = 0.0
    return np.asarray(rslt)


def euDYN_rotm(float[:, ::1] R, float[::1] eu):
    cdef float cos_theta = cos(eu[2])
    cdef float cos_phi   = cos(eu[1])
    cdef float cos_psi   = cos(eu[0])
    cdef float sin_theta = sin(eu[2])
    cdef float sin_phi   = sin(eu[1])
    cdef float sin_psi   = sin(eu[0])
    R[0, 0] =  cos_theta*cos_psi - cos_phi*sin_theta*sin_psi
    R[0, 1] = -cos_theta*sin_psi - cos_phi*cos_psi*sin_theta
    R[0, 2] =  sin_theta*sin_phi
    R[1, 0] =  cos_psi*sin_theta + cos_theta*cos_phi*sin_psi
    R[1, 1] =  cos_theta*cos_phi*cos_psi - sin_theta*sin_psi
    R[1, 2] = -cos_theta*sin_phi
    R[2, 0] =  sin_phi*sin_psi
    R[2, 1] =  cos_psi*sin_phi
    R[2, 2] =  cos_phi


def euZYZ_rotm(float[:, ::1] R, float[::1] eu):
    cdef float cos_theta = cos(eu[0])
    cdef float cos_phi   = cos(eu[1])
    cdef float cos_psi   = cos(eu[2])
    cdef float sin_theta = sin(eu[0])
    cdef float sin_phi   = sin(eu[1])
    cdef float sin_psi   = sin(eu[2])
    R[0, 0] =  cos_theta*cos_phi*cos_psi - sin_theta*sin_psi
    R[0, 1] = -cos_psi*sin_theta - cos_theta*cos_phi*sin_psi
    R[0, 2] =  cos_theta*sin_phi
    R[1, 0] =  cos_theta*sin_psi + cos_phi*cos_psi*sin_theta
    R[1, 1] =  cos_theta*cos_psi - cos_phi*sin_theta*sin_psi
    R[1, 2] =  sin_theta*sin_phi
    R[2, 0] = -cos_psi*sin_phi
    R[2, 1] =  sin_phi*sin_psi
    R[2, 2] =  cos_phi


def rotm_euZYZ(float[::1] euler, float[:, ::1] R):
    if fabs(R[2, 2] - 1) < 1e-6:
        euler[0] = 0.0
        euler[1] = 0.0
        euler[2] = <float>atan2(R[1, 0], R[1, 1])
    elif fabs(R[2, 2] + 1) < 1e-6:
        euler[0] = 0.0
        euler[1] = 3.14159265358979323846
        euler[2] = <float>atan2(R[1, 0], R[1, 1])
    else:
        euler[0] = <float>atan2(R[1, 2], R[0, 2])
        euler[1] = <float>atan2(sqrt(fabs(1.0 - R[2,2]*R[2,2])), R[2, 2])
        euler[2] = <float>atan2(R[2, 1], -R[2, 0])


cdef inline int _ceil_div(int N, double scale) noexcept nogil:
    return <int>ceil(<double>N / scale)


cdef int _build_axis_table(int N, double scale, int N_b, double offset,
                           int max_k,
                           int* counts, int* starts, float* weights) noexcept nogil:
    """Populate per-output-pixel input ranges and weights along one axis.

    For each output pixel j, fills:
      starts[j]  : first input index (may be negative; entries below 0 are skipped)
      counts[j]  : number of in-bounds input pixels contributing
      weights[j*max_k + k] : overlap weight for the k-th contributing input pixel

    The k-th contributing input pixel is at index (starts[j] + skipped_lo + k),
    where skipped_lo is the count of leading OOB indices. To keep the inner
    loop branch-free we shift `starts[j]` to the first in-bounds index and
    skip the OOB partial weights at the boundaries.
    """
    cdef int j, i, i_first, i_last
    cdef double a, b, left, right, w
    cdef int idx_w
    cdef int k, n_inb
    cdef int first_inb_offset

    for j in range(N_b):
        a = offset + j * scale
        b = a + scale
        i_first = <int>floor(a)
        i_last  = <int>ceil(b) - 1
        n_inb = 0
        first_inb_offset = -1
        idx_w = j * max_k
        for k in range(max_k):
            weights[idx_w + k] = 0.0
        for i in range(i_first, i_last + 1):
            if i < 0 or i >= N:
                continue
            left  = <double>i
            if a > left:
                left = a
            right = <double>(i + 1)
            if b < right:
                right = b
            w = right - left
            if w <= 0.0:
                continue
            if first_inb_offset < 0:
                first_inb_offset = i
            weights[idx_w + n_inb] = <float>w
            n_inb += 1
        if first_inb_offset < 0:
            starts[j] = 0
        else:
            starts[j] = first_inb_offset
        counts[j] = n_inb
    return 0


def bin_frame(float[:, ::1] out_frame, float[:, ::1] in_frame, double scale):
    """Area-weighted downsample of a single 2-D frame.

    Output size follows ``ceil`` of the input divided by ``scale``. The
    window offset ``(N - N_b*scale)/2 - (scale-1)/2`` keeps the sampling
    origin on SUSAN's pixel-centre convention, where input index ``i`` sits
    at coordinate ``i`` and the tomogram centre is at ``stk_center = N/2``.
    An extended particle therefore projects to the same physical point at
    every binning level. (The ``-(scale-1)/2`` term matters: without it, the
    geometric box-edge centre is preserved instead, which shifts binned
    content by ``(scale-1)/2`` input pixels relative to the projector and
    smears the reconstruction across tilts.)
    Edge bins extend past the input boundary; out-of-bounds input pixels
    are skipped and the per-output-pixel weight buffer normalises the result.

    Parameters
    ----------
    out_frame : ndarray (H_b, W_b) float32, contiguous
        Pre-allocated output. Must have shape
        ``(ceil(H/scale), ceil(W/scale))``.
    in_frame : ndarray (H, W) float32, contiguous
    scale : float, > 1.0
    """
    cdef int H = in_frame.shape[0]
    cdef int W = in_frame.shape[1]
    cdef int H_b = out_frame.shape[0]
    cdef int W_b = out_frame.shape[1]
    cdef int H_b_expected = _ceil_div(H, scale)
    cdef int W_b_expected = _ceil_div(W, scale)
    if H_b != H_b_expected or W_b != W_b_expected:
        raise ValueError(
            "out_frame shape (%d, %d) does not match expected (%d, %d) for "
            "in_frame (%d, %d) and scale %f"
            % (H_b, W_b, H_b_expected, W_b_expected, H, W, scale)
        )
    if scale <= 1.0:
        raise ValueError("scale must be > 1.0")

    cdef double offset_y = (H - H_b * scale) / 2.0 - (scale - 1.0) / 2.0
    cdef double offset_x = (W - W_b * scale) / 2.0 - (scale - 1.0) / 2.0
    cdef int max_k = <int>ceil(scale) + 1

    cdef int* row_counts  = <int*>  malloc(H_b * sizeof(int))
    cdef int* row_starts  = <int*>  malloc(H_b * sizeof(int))
    cdef float* row_wgt   = <float*>malloc(H_b * max_k * sizeof(float))
    cdef int* col_counts  = <int*>  malloc(W_b * sizeof(int))
    cdef int* col_starts  = <int*>  malloc(W_b * sizeof(int))
    cdef float* col_wgt   = <float*>malloc(W_b * max_k * sizeof(float))
    cdef float* out_line  = <float*>malloc(W_b * sizeof(float))
    cdef float* wgt_line  = <float*>malloc(W_b * sizeof(float))

    if (row_counts == NULL or row_starts == NULL or row_wgt == NULL
        or col_counts == NULL or col_starts == NULL or col_wgt == NULL
        or out_line == NULL or wgt_line == NULL):
        free(row_counts); free(row_starts); free(row_wgt)
        free(col_counts); free(col_starts); free(col_wgt)
        free(out_line);   free(wgt_line)
        raise MemoryError()

    cdef int j_row, j_col, k_row, k_col, i_row, i_col
    cdef float w_y, w_xy, pix_val
    cdef int row_base, col_base

    try:
        with nogil:
            _build_axis_table(H, scale, H_b, offset_y, max_k,
                              row_counts, row_starts, row_wgt)
            _build_axis_table(W, scale, W_b, offset_x, max_k,
                              col_counts, col_starts, col_wgt)

            for j_row in range(H_b):
                for j_col in range(W_b):
                    out_line[j_col] = 0.0
                    wgt_line[j_col] = 0.0

                row_base = j_row * max_k
                for k_row in range(row_counts[j_row]):
                    i_row = row_starts[j_row] + k_row
                    w_y   = row_wgt[row_base + k_row]
                    for j_col in range(W_b):
                        col_base = j_col * max_k
                        for k_col in range(col_counts[j_col]):
                            i_col = col_starts[j_col] + k_col
                            w_xy  = w_y * col_wgt[col_base + k_col]
                            out_line[j_col] += in_frame[i_row, i_col] * w_xy
                            wgt_line[j_col] += w_xy

                for j_col in range(W_b):
                    if wgt_line[j_col] > 0.0:
                        out_frame[j_row, j_col] = out_line[j_col] / wgt_line[j_col]
                    else:
                        out_frame[j_row, j_col] = 0.0
    finally:
        free(row_counts); free(row_starts); free(row_wgt)
        free(col_counts); free(col_starts); free(col_wgt)
        free(out_line);   free(wgt_line)


def bin_frame_shape(int H, int W, double scale):
    """Return the (H_b, W_b) output shape ``bin_frame`` would produce."""
    if scale <= 1.0:
        raise ValueError("scale must be > 1.0")
    return (<int>ceil(<double>H / scale), <int>ceil(<double>W / scale))
