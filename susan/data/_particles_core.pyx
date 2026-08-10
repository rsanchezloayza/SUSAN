# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np

from libc.stdio  cimport FILE, fopen, fclose, fread, fwrite, fseek, SEEK_SET
from libc.stdlib cimport malloc, free


def _load_all(bytes filename,
              int n_ptcl, int n_refs, int n_proj,
              unsigned int[::1]  ptcl_id,
              unsigned int[::1]  tomo_id,
              unsigned int[::1]  tomo_cix,
              float[:, ::1]      pos,
              unsigned int[::1]  ref_cix,
              unsigned int[::1]  half_id,
              float[::1]         extra_1,
              float[::1]         extra_2,
              float[:, :, ::1]   ali_eu,
              float[:, :, ::1]   ali_t,
              float[:, ::1]      ali_cc,
              float[:, ::1]      ali_w,
              float[:, :, ::1]   prj_eu,
              float[:, :, ::1]   prj_t,
              float[:, ::1]      prj_cc,
              float[:, ::1]      prj_w,
              float[:, ::1]      def_U,
              float[:, ::1]      def_V,
              float[:, ::1]      def_ang,
              float[:, ::1]      def_phas,
              float[:, ::1]      def_Bfct,
              float[:, ::1]      def_ExFl,
              float[:, ::1]      def_mres,
              float[:, ::1]      def_scor):

    cdef int floats_per_ptcl = 10 + 8*n_refs + 15*n_proj
    cdef float         *buf  = <float *>malloc(floats_per_ptcl * sizeof(float))
    if buf == NULL:
        raise MemoryError()
    cdef unsigned int  *ubuf = <unsigned int *>buf

    cdef FILE *fp = fopen(filename, b"rb")
    if fp == NULL:
        free(buf)
        raise IOError("Cannot open file")

    # Skip 20-byte header: 8 magic + 3×uint32
    fseek(fp, 20, SEEK_SET)

    cdef int i, j, o
    for i in range(n_ptcl):
        fread(buf, sizeof(float), floats_per_ptcl, fp)

        # stg1
        ptcl_id [i]    = ubuf[0]
        tomo_id [i]    = ubuf[1]
        tomo_cix[i]    = ubuf[2]
        pos     [i, 0] = buf [3]
        pos     [i, 1] = buf [4]
        pos     [i, 2] = buf [5]
        ref_cix [i]    = ubuf[6]
        half_id [i]    = ubuf[7]
        extra_1 [i]    = buf [8]
        extra_2 [i]    = buf [9]

        # stg2: 3-D alignment
        o = 10
        for j in range(n_refs):
            ali_eu[j, i, 0] = buf[o]; ali_eu[j, i, 1] = buf[o+1]; ali_eu[j, i, 2] = buf[o+2]
            o += 3
        for j in range(n_refs):
            ali_t [j, i, 0] = buf[o]; ali_t [j, i, 1] = buf[o+1]; ali_t [j, i, 2] = buf[o+2]
            o += 3
        for j in range(n_refs):
            ali_cc[j, i] = buf[o]; o += 1
        for j in range(n_refs):
            ali_w [j, i] = buf[o]; o += 1

        # stg3: 2-D alignment
        for j in range(n_proj):
            prj_eu[i, j, 0] = buf[o]; prj_eu[i, j, 1] = buf[o+1]; prj_eu[i, j, 2] = buf[o+2]
            o += 3
        for j in range(n_proj):
            prj_t [i, j, 0] = buf[o]; prj_t [i, j, 1] = buf[o+1]
            o += 2
        for j in range(n_proj):
            prj_cc[i, j] = buf[o]; o += 1
        for j in range(n_proj):
            prj_w [i, j] = buf[o]; o += 1

        # stg4: CTF
        for j in range(n_proj):
            def_U   [i, j] = buf[o];   def_V   [i, j] = buf[o+1]
            def_ang [i, j] = buf[o+2]; def_phas[i, j] = buf[o+3]
            def_Bfct[i, j] = buf[o+4]; def_ExFl[i, j] = buf[o+5]
            def_mres[i, j] = buf[o+6]; def_scor[i, j] = buf[o+7]
            o += 8

    fclose(fp)
    free(buf)


def _save_all(bytes filename,
              int n_ptcl, int n_refs, int n_proj,
              unsigned int[::1]  ptcl_id,
              unsigned int[::1]  tomo_id,
              unsigned int[::1]  tomo_cix,
              float[:, ::1]      pos,
              unsigned int[::1]  ref_cix,
              unsigned int[::1]  half_id,
              float[::1]         extra_1,
              float[::1]         extra_2,
              float[:, :, ::1]   ali_eu,
              float[:, :, ::1]   ali_t,
              float[:, ::1]      ali_cc,
              float[:, ::1]      ali_w,
              float[:, :, ::1]   prj_eu,
              float[:, :, ::1]   prj_t,
              float[:, ::1]      prj_cc,
              float[:, ::1]      prj_w,
              float[:, ::1]      def_U,
              float[:, ::1]      def_V,
              float[:, ::1]      def_ang,
              float[:, ::1]      def_phas,
              float[:, ::1]      def_Bfct,
              float[:, ::1]      def_ExFl,
              float[:, ::1]      def_mres,
              float[:, ::1]      def_scor):

    cdef int floats_per_ptcl = 10 + 8*n_refs + 15*n_proj
    cdef float        *buf   = <float *>malloc(floats_per_ptcl * sizeof(float))
    if buf == NULL:
        raise MemoryError()
    cdef unsigned int *ubuf  = <unsigned int *>buf

    cdef FILE *fp = fopen(filename, b"wb")
    if fp == NULL:
        free(buf)
        raise IOError("Cannot open file for writing")

    # Header: magic + n_ptcl, n_proj, n_refs
    cdef unsigned int hdr[3]
    hdr[0] = n_ptcl; hdr[1] = n_proj; hdr[2] = n_refs
    fwrite(<char *>b"SsaPtcl1", 1, 8, fp)
    fwrite(hdr, sizeof(unsigned int), 3, fp)

    cdef int i, j, o
    for i in range(n_ptcl):

        # stg1
        ubuf[0] = ptcl_id [i]
        ubuf[1] = tomo_id [i]
        ubuf[2] = tomo_cix[i]
        buf [3] = pos     [i, 0]
        buf [4] = pos     [i, 1]
        buf [5] = pos     [i, 2]
        ubuf[6] = ref_cix [i]
        ubuf[7] = half_id [i]
        buf [8] = extra_1 [i]
        buf [9] = extra_2 [i]

        # stg2: 3-D alignment
        o = 10
        for j in range(n_refs):
            buf[o] = ali_eu[j, i, 0]; buf[o+1] = ali_eu[j, i, 1]; buf[o+2] = ali_eu[j, i, 2]
            o += 3
        for j in range(n_refs):
            buf[o] = ali_t [j, i, 0]; buf[o+1] = ali_t [j, i, 1]; buf[o+2] = ali_t [j, i, 2]
            o += 3
        for j in range(n_refs):
            buf[o] = ali_cc[j, i]; o += 1
        for j in range(n_refs):
            buf[o] = ali_w [j, i]; o += 1

        # stg3: 2-D alignment
        for j in range(n_proj):
            buf[o] = prj_eu[i, j, 0]; buf[o+1] = prj_eu[i, j, 1]; buf[o+2] = prj_eu[i, j, 2]
            o += 3
        for j in range(n_proj):
            buf[o] = prj_t [i, j, 0]; buf[o+1] = prj_t [i, j, 1]
            o += 2
        for j in range(n_proj):
            buf[o] = prj_cc[i, j]; o += 1
        for j in range(n_proj):
            buf[o] = prj_w [i, j]; o += 1

        # stg4: CTF
        for j in range(n_proj):
            buf[o]   = def_U   [i, j]; buf[o+1] = def_V   [i, j]
            buf[o+2] = def_ang [i, j]; buf[o+3] = def_phas[i, j]
            buf[o+4] = def_Bfct[i, j]; buf[o+5] = def_ExFl[i, j]
            buf[o+6] = def_mres[i, j]; buf[o+7] = def_scor[i, j]
            o += 8

        fwrite(buf, sizeof(float), floats_per_ptcl, fp)

    fclose(fp)
    free(buf)


def _update_new_defocus(float[::1] dU, float[::1] dV,
                        float[:, :, ::1] R, float[::1] p,
                        int num_proj, float z_coef,
                        float[::1] dU_in, float[::1] dV_in):
    cdef int i
    cdef float dZ
    for i in range(num_proj):
        dZ    = R[i, 2, 0]*p[0] + R[i, 2, 1]*p[1] + R[i, 2, 2]*p[2]
        dU[i] = dU_in[i] + z_coef * dZ
        dV[i] = dV_in[i] + z_coef * dZ
