# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport cos, sin, atan2, sqrt, fabs


# ── Inline helpers (mirroring euZYZ_rotm / rotm_euZYZ from _functions_core) ──

cdef inline void _euZYZ_rotm(float[:, ::1] R, float[::1] eu) noexcept nogil:
    cdef float ct = cos(eu[0]), cp = cos(eu[1]), cps = cos(eu[2])
    cdef float st = sin(eu[0]), sp = sin(eu[1]), sps = sin(eu[2])
    R[0, 0] =  ct*cp*cps - st*sps
    R[0, 1] = -cps*st - ct*cp*sps
    R[0, 2] =  ct*sp
    R[1, 0] =  ct*sps + cp*cps*st
    R[1, 1] =  ct*cps - cp*st*sps
    R[1, 2] =  st*sp
    R[2, 0] = -cps*sp
    R[2, 1] =  sp*sps
    R[2, 2] =  cp


cdef inline void _rotm_euZYZ(float[::1] euler, float[:, ::1] R) noexcept nogil:
    if fabs(R[2, 2] - 1.0) < 1e-6:
        euler[0] = 0.0
        euler[1] = 0.0
        euler[2] = <float>atan2(R[1, 0], R[1, 1])
    elif fabs(R[2, 2] + 1.0) < 1e-6:
        euler[0] = 0.0
        euler[1] = 3.14159265358979323846
        euler[2] = <float>atan2(R[1, 0], R[1, 1])
    else:
        euler[0] = <float>atan2(R[1, 2], R[0, 2])
        euler[1] = <float>atan2(sqrt(fabs(1.0 - R[2,2]*R[2,2])), R[2, 2])
        euler[2] = <float>atan2(R[2, 1], -R[2, 0])


# ── Public functions ──────────────────────────────────────────────────────────

def _inplace_shift(float[:, ::1] ali_eZYZ, float[:, ::1] ali_t, float[::1] t):
    cdef int n = ali_eZYZ.shape[0], i
    cdef float R[3][3]
    cdef float[:, ::1] Rv = <float[:3, :3]>&R[0][0]
    cdef float t0, t1, t2

    for i in range(n):
        _euZYZ_rotm(Rv, ali_eZYZ[i])
        t0 = R[0][0]*t[0] + R[0][1]*t[1] + R[0][2]*t[2]
        t1 = R[1][0]*t[0] + R[1][1]*t[1] + R[1][2]*t[2]
        t2 = R[2][0]*t[0] + R[2][1]*t[1] + R[2][2]*t[2]
        ali_t[i, 0] += t0
        ali_t[i, 1] += t1
        ali_t[i, 2] += t2


def _inplace_rot_shift(float[:, ::1] ali_eZYZ, float[:, ::1] ali_t,
                       float[:, ::1] R, float[::1] t):
    cdef int n = ali_eZYZ.shape[0], i, a, b, k
    cdef float R_in[3][3]
    cdef float Rout[3][3]
    cdef float[:, ::1] R_inv = <float[:3, :3]>&R_in[0][0]
    cdef float[:, ::1] R_out = <float[:3, :3]>&Rout[0][0]
    cdef float t0, t1, t2

    for i in range(n):
        _euZYZ_rotm(R_inv, ali_eZYZ[i])
        # Rout = R_in @ R^T  (equivalent to (R @ R_in^T)^T)
        for a in range(3):
            for b in range(3):
                Rout[a][b] = 0.0
                for k in range(3):
                    Rout[a][b] += R_in[a][k] * R[b, k]
        _rotm_euZYZ(ali_eZYZ[i], R_out)
        t0 = Rout[0][0]*t[0] + Rout[0][1]*t[1] + Rout[0][2]*t[2]
        t1 = Rout[1][0]*t[0] + Rout[1][1]*t[1] + Rout[1][2]*t[2]
        t2 = Rout[2][0]*t[0] + Rout[2][1]*t[1] + Rout[2][2]*t[2]
        ali_t[i, 0] += t0
        ali_t[i, 1] += t1
        ali_t[i, 2] += t2


def _outplace_shift(float[:, ::1] out_ali_t,
                    float[:, ::1] in_ali_eZYZ, float[:, ::1] in_ali_t,
                    float[:, ::1] t):
    cdef int n_ptcl = in_ali_eZYZ.shape[0], n_t = t.shape[0]
    cdef int i, j, out_ix = 0
    cdef float R[3][3]
    cdef float[:, ::1] Rv = <float[:3, :3]>&R[0][0]
    cdef float t0, t1, t2

    for i in range(n_ptcl):
        _euZYZ_rotm(Rv, in_ali_eZYZ[i])
        for j in range(n_t):
            t0 = R[0][0]*t[j,0] + R[0][1]*t[j,1] + R[0][2]*t[j,2]
            t1 = R[1][0]*t[j,0] + R[1][1]*t[j,1] + R[1][2]*t[j,2]
            t2 = R[2][0]*t[j,0] + R[2][1]*t[j,1] + R[2][2]*t[j,2]
            out_ali_t[out_ix, 0] = in_ali_t[i, 0] + t0
            out_ali_t[out_ix, 1] = in_ali_t[i, 1] + t1
            out_ali_t[out_ix, 2] = in_ali_t[i, 2] + t2
            out_ix += 1


def _outplace_rot_shift(float[:, ::1] out_ali_eZYZ, float[:, ::1] out_ali_t,
                        float[:, ::1] in_ali_eZYZ, float[:, ::1] in_ali_t,
                        float[:, :, ::1] R, float[:, ::1] t):
    cdef int n_ptcl = in_ali_eZYZ.shape[0], n_t = t.shape[0]
    cdef int i, j, a, b, k, out_ix = 0
    cdef float R_in[3][3]
    cdef float Rout[3][3]
    cdef float[:, ::1] R_inv = <float[:3, :3]>&R_in[0][0]
    cdef float[:, ::1] R_out = <float[:3, :3]>&Rout[0][0]
    cdef float t0, t1, t2

    for i in range(n_ptcl):
        _euZYZ_rotm(R_inv, in_ali_eZYZ[i])
        for j in range(n_t):
            # Rout = R_in @ R[j]^T
            for a in range(3):
                for b in range(3):
                    Rout[a][b] = 0.0
                    for k in range(3):
                        Rout[a][b] += R_in[a][k] * R[j, b, k]
            t0 = Rout[0][0]*t[j,0] + Rout[0][1]*t[j,1] + Rout[0][2]*t[j,2]
            t1 = Rout[1][0]*t[j,0] + Rout[1][1]*t[j,1] + Rout[1][2]*t[j,2]
            t2 = Rout[2][0]*t[j,0] + Rout[2][1]*t[j,1] + Rout[2][2]*t[j,2]
            out_ali_t[out_ix, 0] = in_ali_t[i, 0] + t0
            out_ali_t[out_ix, 1] = in_ali_t[i, 1] + t1
            out_ali_t[out_ix, 2] = in_ali_t[i, 2] + t2
            _rotm_euZYZ(out_ali_eZYZ[out_ix], R_out)
            out_ix += 1


def _enable_by_tilt(float[:, ::1] prj_w, unsigned int[::1] tomo_cix,
                    float[:, :, ::1] prj_eu, float[:, ::1] prj_wgt,
                    float tilt_min, float tilt_max):
    cdef int n_ptcl = prj_w.shape[0], n_proj = prj_w.shape[1]
    cdef int p, k, t_id
    cdef float tilt

    for p in range(n_ptcl):
        t_id = tomo_cix[p]
        for k in range(n_proj):
            if prj_wgt[t_id, k] > 0:
                tilt = fabs(prj_eu[t_id, k, 1])
                prj_w[p, k] = 1.0 if (tilt < tilt_max and tilt >= tilt_min) else 0.0
            else:
                prj_w[p, k] = 0.0


def _enable_by_tilt_range(float[:, ::1] prj_w, unsigned int[::1] tomo_cix,
                           float[:, :, ::1] prj_eu, float[:, ::1] prj_wgt,
                           float tilt_min, float tilt_max):
    cdef int n_ptcl = prj_w.shape[0], n_proj = prj_w.shape[1]
    cdef int p, k, t_id
    cdef float eu0, eu1, eu2, c0, s0, horiz, tilt
    cdef float R[3][3]
    cdef float eu_tmp[3]
    cdef float[:, ::1] Rv   = <float[:3, :3]>&R[0][0]
    cdef float[::1]    eutv = <float[:3]>&eu_tmp[0]
    cdef float PI = 3.14159265358979323846

    for p in range(n_ptcl):
        t_id = tomo_cix[p]
        for k in range(n_proj):
            if prj_wgt[t_id, k] > 0:
                eu_tmp[0] = prj_eu[t_id, k, 0] * PI / 180.0
                eu_tmp[1] = prj_eu[t_id, k, 1] * PI / 180.0
                eu_tmp[2] = prj_eu[t_id, k, 2] * PI / 180.0
                _euZYZ_rotm(Rv, eutv)
                c0    = cos(eu_tmp[0])
                s0    = sin(eu_tmp[0])
                horiz = c0*R[0][2] + s0*R[1][2]
                tilt  = <float>atan2(horiz, R[2][2])
                prj_w[p, k] = 1.0 if (tilt >= tilt_min and tilt < tilt_max) else 0.0
            else:
                prj_w[p, k] = 0.0


def _disable_closer(unsigned char[::1] w_mask, float[:, ::1] pos,
                    long[::1] sort_ix, float dist2):
    cdef int N = sort_ix.shape[0], i_cur, i_nxt
    cdef long idx_cur, idx_nxt
    cdef float dx, dy, dz, d

    for i_cur in range(N):
        idx_cur = sort_ix[i_cur]
        if w_mask[idx_cur]:
            for i_nxt in range(i_cur + 1, N):
                idx_nxt = sort_ix[i_nxt]
                if w_mask[idx_nxt]:
                    dx = pos[idx_cur, 0] - pos[idx_nxt, 0]
                    dy = pos[idx_cur, 1] - pos[idx_nxt, 1]
                    dz = pos[idx_cur, 2] - pos[idx_nxt, 2]
                    d = dx*dx + dy*dy + dz*dz
                    if d < dist2:
                        w_mask[idx_nxt] = 0


def _get_min_dist(float[::1] dist, float[:, ::1] pos):
    cdef int N = pos.shape[0], idx_cur, idx_wrk
    cdef float dx, dy, dz, d, min_d

    for idx_cur in range(N):
        min_d = 1e30
        for idx_wrk in range(N):
            if idx_cur != idx_wrk:
                dx = pos[idx_cur, 0] - pos[idx_wrk, 0]
                dy = pos[idx_cur, 1] - pos[idx_wrk, 1]
                dz = pos[idx_cur, 2] - pos[idx_wrk, 2]
                d = sqrt(dx*dx + dy*dy + dz*dz)
                if d < min_d:
                    min_d = d
        dist[idx_cur] = min_d
