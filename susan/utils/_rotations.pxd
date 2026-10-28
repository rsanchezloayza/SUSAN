# cython: language_level=3
# Rotation kernels shared by the Cython cores. R is row-major (9 values).
# euZYZ: R = Rz(eu0) Ry(eu1) Rz(eu2). euDYN: R = Rz(eu2) Rx(eu1) Rz(eu0).

from libc.math cimport cos, sin, atan2, sqrt, fabs

ctypedef fused real:
    float
    double


cdef inline void euZYZ_R(const real* eu, real* R) noexcept nogil:
    cdef real ct = cos(eu[0]), cp = cos(eu[1]), cs = cos(eu[2])
    cdef real st = sin(eu[0]), sp = sin(eu[1]), ss = sin(eu[2])
    R[0] =  ct*cp*cs - st*ss
    R[1] = -cs*st - ct*cp*ss
    R[2] =  ct*sp
    R[3] =  ct*ss + cp*cs*st
    R[4] =  ct*cs - cp*st*ss
    R[5] =  st*sp
    R[6] = -cs*sp
    R[7] =  sp*ss
    R[8] =  cp


cdef inline void R_euZYZ(const real* R, real* eu) noexcept nogil:
    if fabs(R[8] - 1) < 1e-6:
        eu[0] = 0.0
        eu[1] = 0.0
        eu[2] = <real>atan2(R[3], R[4])
    elif fabs(R[8] + 1) < 1e-6:
        eu[0] = 0.0
        eu[1] = <real>3.14159265358979323846
        eu[2] = <real>atan2(R[3], R[4])
    else:
        eu[0] = <real>atan2(R[5], R[2])
        eu[1] = <real>atan2(sqrt(fabs(1.0 - R[8]*R[8])), R[8])
        eu[2] = <real>atan2(R[7], -R[6])


cdef inline void euDYN_R(const real* eu, real* R) noexcept nogil:
    cdef real ct = cos(eu[2]), cp = cos(eu[1]), cs = cos(eu[0])
    cdef real st = sin(eu[2]), sp = sin(eu[1]), ss = sin(eu[0])
    R[0] =  ct*cs - cp*st*ss
    R[1] = -ct*ss - cp*cs*st
    R[2] =  st*sp
    R[3] =  cs*st + ct*cp*ss
    R[4] =  ct*cp*cs - st*ss
    R[5] = -ct*sp
    R[6] =  sp*ss
    R[7] =  cs*sp
    R[8] =  cp
