/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
 * Max Planck Institute of Biophysics
 * Department of Structural Biology - Kudryashev Group.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef MATH_CPU_H
#define MATH_CPU_H

#if __CUDACC_VER_MAJOR__ == 12
    #pragma nv_diag_suppress 20012
    #pragma diag_suppress 20012
#elif __CUDACC_VER_MAJOR__ == 11
    #pragma nv_diag_suppress 20011
    #pragma nv_diag_suppress 20012
    #pragma nv_diag_suppress 20014
    #pragma nv_diag_suppress 20236
    #pragma diag_suppress 20012
    #pragma diag_suppress 3123
    #pragma diag_suppress 3124
    #pragma diag_suppress 3126
#elif __CUDACC_VER_MAJOR__ == 10
    #pragma diag_suppress 2976
    #pragma diag_suppress 2979
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>
#include <sys/time.h>
#include <sys/types.h>

#include "iostream"

#include "datatypes.h"
#include "Eigen/Geometry"

typedef Eigen::Matrix3f M33f;
typedef Eigen::Vector3f V3f;

using namespace Eigen;

namespace Math {

inline float get_lambda(const float kv) {
    float volt = kv*1000;
    return sqrtf( 150.4f / ( volt*(1.0f+(volt/1022000.0f)) ) );
}

inline bool should_use_avx2(const uint32 length) {
    return (__builtin_cpu_supports ("avx2") && ( (length&31) == 0));
}

inline void eZXZ_Rmat(M33f&R,const V3f&eu_rad) {
    R = AngleAxisf(eu_rad(0), Vector3f::UnitZ())
      * AngleAxisf(eu_rad(1), Vector3f::UnitX())
      * AngleAxisf(eu_rad(2), Vector3f::UnitZ());
}

inline void eZYZ_Rmat(M33f&R,const V3f&eu_rad) {
    R = AngleAxisf(eu_rad(0), Vector3f::UnitZ())
      * AngleAxisf(eu_rad(1), Vector3f::UnitY())
      * AngleAxisf(eu_rad(2), Vector3f::UnitZ());
}

inline void eXYZ_Rmat(M33f&R,const V3f&eu_rad) {
    R = AngleAxisf(eu_rad(0), Vector3f::UnitX())
      * AngleAxisf(eu_rad(1), Vector3f::UnitY())
      * AngleAxisf(eu_rad(2), Vector3f::UnitZ());
}

inline void eZXZ_Rmat(M33f&R,const Vec3&eu_rad) {
    R = AngleAxisf(eu_rad.x, Vector3f::UnitZ())
      * AngleAxisf(eu_rad.y, Vector3f::UnitX())
      * AngleAxisf(eu_rad.z, Vector3f::UnitZ());
}

inline void eZYZ_Rmat(M33f&R,const Vec3&eu_rad) {
    R = AngleAxisf(eu_rad.x, Vector3f::UnitZ())
      * AngleAxisf(eu_rad.y, Vector3f::UnitY())
      * AngleAxisf(eu_rad.z, Vector3f::UnitZ());
}

inline void eXYZ_Rmat(M33f&R,const Vec3&eu_rad) {
    R = AngleAxisf(eu_rad.x, Vector3f::UnitX())
      * AngleAxisf(eu_rad.y, Vector3f::UnitY())
      * AngleAxisf(eu_rad.z, Vector3f::UnitZ());
}

inline void Rmat_eZXZ(V3f&eu_rad,const M33f&R) {
    // Eigen gives the correct angles, but we want to make sure they are in the expected ranges.
    // R = Rz(eu0)Rx(eu1)Rz(eu2),  eu0,eu2 in [-pi,pi], eu1 in [0,pi].
    // At gimbal lock (|sin(eu1)| ~ 0), eu0 and eu2 are not separately
    // determined; we set eu0 = 0 and put the residual into eu2.
    const float sb = sqrt(R(0,2)*R(0,2) + R(1,2)*R(1,2));   // |sin(eu1)|

    if( sb > 1e-6f ) {
        eu_rad(0) = atan2( R(0,2), -R(1,2));
        eu_rad(1) = atan2( sb,      R(2,2));
        eu_rad(2) = atan2( R(2,0),  R(2,1));
    }
    else if( R(2,2) > 0.0f ) {     // eu1 = 0:  R = Rz(eu0+eu2)
        eu_rad(0) = 0.0f;
        eu_rad(1) = 0.0f;
        eu_rad(2) = atan2(R(1,0), R(0,0));
    }
    else {                          // eu1 = pi: R = Rx(pi)Rz(eu2-eu0)
        eu_rad(0) = 0.0f;
        eu_rad(1) = float(M_PI);
        eu_rad(2) = atan2(-R(0,1), R(0,0));
    }
}

inline void Rmat_eZYZ(V3f&eu_rad,const M33f&R) {
    // Eigen gives the correct angles, but we want to make sure they are in the expected ranges.
    // R = Rz(eu0)Ry(eu1)Rz(eu2),  eu0,eu2 in [-pi,pi], eu1 in [0,pi].
    // At gimbal lock (|sin(eu1)| ~ 0), eu0 and eu2 are not separately
    // determined; we set eu0 = 0 and put the residual into eu2.
    const float sb = sqrt(R(0,2)*R(0,2) + R(1,2)*R(1,2));   // |sin(eu1)|

    if( sb > 1e-6f ) {
        eu_rad(0) = atan2( R(1,2),  R(0,2));
        eu_rad(1) = atan2( sb,      R(2,2));
        eu_rad(2) = atan2( R(2,1), -R(2,0));
    }
    else if( R(2,2) > 0.0f ) {     // eu1 = 0:  R = Rz(eu0+eu2)
        eu_rad(0) = 0.0f;
        eu_rad(1) = 0.0f;
        eu_rad(2) = atan2(R(1,0), R(0,0));
    }
    else {                          // eu1 = pi: R = Ry(pi)Rz(eu2-eu0)
        eu_rad(0) = 0.0f;
        eu_rad(1) = float(M_PI);
        eu_rad(2) = atan2(R(0,1), R(1,1));
    }
}

inline void Rmat_eZXZ(Vec3&eu_rad,const M33f&R) {
    V3f tmp;
    Rmat_eZXZ(tmp,R);
    eu_rad.x = tmp(0);
    eu_rad.y = tmp(1);
    eu_rad.z = tmp(2);
}

inline void Rmat_eZYZ(Vec3&eu_rad,const M33f&R) {
    V3f tmp;
    Rmat_eZYZ(tmp,R);
    eu_rad.x = tmp(0);
    eu_rad.y = tmp(1);
    eu_rad.z = tmp(2);
}

inline void set(Rot33&Rout,const M33f&Rin) {
    Rout.xx = Rin(0,0);
    Rout.xy = Rin(0,1);
    Rout.xz = Rin(0,2);
    Rout.yx = Rin(1,0);
    Rout.yy = Rin(1,1);
    Rout.yz = Rin(1,2);
    Rout.zx = Rin(2,0);
    Rout.zy = Rin(2,1);
    Rout.zz = Rin(2,2);
}

inline void set(M33f&Rout,const Rot33&Rin) {
    Rout(0,0) = Rin.xx;
    Rout(0,1) = Rin.xy;
    Rout(0,2) = Rin.xz;
    Rout(1,0) = Rin.yx;
    Rout(1,1) = Rin.yy;
    Rout(1,2) = Rin.yz;
    Rout(2,0) = Rin.zx;
    Rout(2,1) = Rin.zy;
    Rout(2,2) = Rin.zz;
}

inline float get_Y_angle_rad(const M33f&R) {
    V3f p_z(0.0,0.0,1.0);
    V3f p = R*p_z;
    float l = sqrtf( p(1)*p(1) + p(2)*p(2) );
    return atan2( p(0), l );
}

inline void post_transposed_rotation(Rot33& R_work, const Rot33& Rot) {
    Rot33 result;

    const float*A = &(R_work.xx);
    const float*B = &(Rot.xx);
          float*C = &(result.xx);

    for(int y=0;y<3;y++) {
        for (int x=0;x<3;x++) {
            C[y*3+x]  = A[y*3  ] * B[x*3  ];
            C[y*3+x] += A[y*3+1] * B[x*3+1];
            C[y*3+x] += A[y*3+2] * B[x*3+2];
        }
    }
    R_work = result;
}

inline int make_even_up(const float val) {
    return (int)(2*ceil(val/2));
}

inline void sum(float*ptr_out, const float*ptr_in, const uint32 length) {
    uint32 i;
    const float *w_in  = ptr_in;
    float *w_out = ptr_out;

    if( should_use_avx2(length) ) {

        for(i=0;i<length;i+=8) {

            __asm__ __volatile__(
                "vmovups (%0), %%ymm0 \n\t"
                "vaddps  (%1), %%ymm0, %%ymm0 \n\t"
                "vmovups %%ymm0, (%0) \n\t"
            :: "r"(w_out), "r"(w_in) :"memory");

            w_in  += 8;
            w_out += 8;
        }

    }
    else {
        for(i=0;i<length;i++) {
            w_out[i] += w_in[i];
        }
    }
}

inline void sum(double*ptr_out, const double*ptr_in, const uint32 length) {
    uint32 i;
    const double *w_in  = ptr_in;
    double *w_out = ptr_out;

    if( should_use_avx2(length) ) {

        for(i=0;i<length;i+=4) {

            __asm__ __volatile__(
                "vmovupd (%0), %%ymm0 \n\t"
                "vaddpd  (%1), %%ymm0, %%ymm0 \n\t"
                "vmovupd %%ymm0, (%0) \n\t"
            :: "r"(w_out), "r"(w_in) :"memory");

            w_in  += 4;
            w_out += 4;
        }

    }
    else {
        for(i=0;i<length;i++) {
            w_out[i] += w_in[i];
        }
    }
}

#ifdef __NVCC__
inline void sum(double2*ptr_out, const double2*ptr_in, const uint32 length) {
	sum((double*)ptr_out,(const double*)ptr_in,length*2);
}
#endif

inline void sort(float*data,const uint32 length) {
    int i,j;
    float key;
    for(i=1;i<length;i++) {
        key = data[i];
        j = i-1;

        while(j>=0 && data[j]>key) {
            data[j+1] = data[j];
            j--;
        }

        data[j+1] = key;
    }
}

inline void mul(float*out,const float*in,const uint32 length) {

    uint32 i;
    if( should_use_avx2(length) ) {

        const float*ptr_i = in;
        float*ptr_o = out;

        for(i=0;i<length;i+=8) {

            __asm__ __volatile__(
                "vmovups (%0), %%ymm0 \n\t"
                "vmulps  (%1), %%ymm0, %%ymm0 \n\t"
                "vmovups %%ymm0, (%1) \n\t"
            :: "r"(ptr_i),"r"(ptr_o) :"memory");

            ptr_i += 8;
            ptr_o += 8;
        }

    }
    else {
        for(i=0;i<length;i++) {
            out[i] = out[i]*in[i];
        }
    }

}

inline void mul(float*out,const float scale,const uint32 length) {

    uint32 i;
    for(i=0;i<length;i++) {
        out[i] = out[i]*scale;
    }

}

inline float get_max(const float*ptr,const uint32 length) {

    float rslt = ptr[0];

    uint32 i;
    for(i=1;i<length;i++) {
        rslt = fmaxf(ptr[i],rslt);
    }

    return rslt;
}

inline void fit_ellipsoid(float&U,float&V,float&ang,const MatrixXf&points) {
    float scale = sqrtf(2.0f/((float)(points.cols())));
    Eigen::JacobiSVD<MatrixXf> svd(points, Eigen::ComputeThinU | Eigen::ComputeThinV);
    U = scale*svd.singularValues()(0);
    V = scale*svd.singularValues()(1);

    Eigen::Matrix2f tmp;
    float sign_correction = svd.matrixU()(0,0)/svd.matrixU()(1,1);
    tmp << svd.matrixU()(0,0), sign_correction*svd.matrixU()(0,1), svd.matrixU()(1,0), sign_correction*svd.matrixU()(1,1);
    Eigen::Rotation2D<float> rot2d(tmp);
    ang = rot2d.angle();
    if(ang<-M_PI/2) ang += M_PI;
    if(ang> M_PI/2) ang -= M_PI;
}

inline float sum_vec(const float*ptr,const uint32 length) {

    float sum = 0;

    uint32 i;

    if( should_use_avx2(length) ) {

        const float*ptr_w = ptr;
        float sum_arr[8];

        __asm__ __volatile__(
            "vxorps %%ymm0, %%ymm0, %%ymm0\n\t"
        :::"memory");

        for(i=0;i<length;i+=8) {

            __asm__ __volatile__(
                "vaddps (%0), %%ymm0, %%ymm0 \n\t"
            :: "r"(ptr_w) :"memory");

            ptr_w += 8;
        }

        __asm__ __volatile__(
            "vmovups %%ymm0, (%0) \n\t"
        ::"r"(sum_arr):"memory");

        for(i=0;i<8;i++) {
            sum += sum_arr[i];
        }

    }
    else {
        for(i=0;i<length;i++) {
            sum += ptr[i];
        }
    }

    return sum;
}

inline void get_avg_std(float&avg,float&std,const float*ptr,const uint32 length) {
	/// One pass estimation of AVG and STD
	
    uint32 i;
    avg = 0;
    std = 0;
    float numel = (float)length;
    
    if( should_use_avx2(length) ) {

        const float*ptr_w = ptr;
        float tmp_arr[8];
        float inv_numel = 1.0/numel;
        
        __asm__ __volatile__(
            "vbroadcastss (%0), %%ymm4\n\t"
            "vxorps %%ymm0, %%ymm0, %%ymm0\n\t"
            "vxorps %%ymm1, %%ymm1, %%ymm1\n\t"
            :: "r"(&inv_numel) :"memory");

        for(int i=0;i<length;i+=8) {
            __asm__ __volatile__(
                "vmovups (%0), %%ymm2 \n\t"
                "vaddps %%ymm2, %%ymm0, %%ymm0 \n\t"
                "vmulps %%ymm2, %%ymm2, %%ymm3 \n\t"
                "vaddps %%ymm3, %%ymm1, %%ymm1 \n\t"
                :: "r"(ptr_w) :"memory");
            ptr_w += 8;
        }

        __asm__ __volatile__(
            "vdpps $0xF1, %%ymm0, %%ymm4, %%ymm2 \n\t"
            "vdpps $0xF2, %%ymm1, %%ymm4, %%ymm3 \n\t"
            "vaddps %%ymm2, %%ymm3, %%ymm3 \n\t"
            "vmovups %%ymm3, (%0) \n\t"
            :: "r"(tmp_arr):"memory");

        avg = tmp_arr[0]+tmp_arr[4];
        std = tmp_arr[1]+tmp_arr[5] - avg*avg;
        std = sqrtf(fmaxf(std, 0.0f));

    }
    else {
        float M = 0.0f, S = 0.0f;
        for(i=0;i<length;i++) {
            float x = ptr[i];
            float old_M = M;
            M += (x - M) / (float)(i + 1);
            S += (x - old_M) * (x - M);
        }
        avg = M;
        std = sqrtf(S / numel);
    }
}

inline void get_min_max_avg_std(float&min,float&max,float&avg,float&std,const float*ptr,const size_t length) {
    /// One pass estimation of AVG and STD

    size_t i;
    avg = 0;
    std = 0;
    min = 0;
    max = 0;
    float numel = (float)length;

    if( should_use_avx2(length) ) {

        const float*ptr_w = ptr;
        float tmp_min[8];
        float tmp_max[8];
        float tmp_arr[8];
        float inv_numel = 1.0/numel;

        __asm__ __volatile__(
            "vbroadcastss (%0), %%ymm4\n\t"
            "vmovups (%1), %%ymm0 \n\t"
            "vmovups (%1), %%ymm5 \n\t"
            "vmovups (%1), %%ymm6 \n\t"
            "vmulps %%ymm0, %%ymm0, %%ymm1 \n\t"
            :: "r"(&inv_numel),"r"(ptr_w) :"memory");
        ptr_w += 8;

        for(i=8;i<length;i+=8) {
            __asm__ __volatile__(
                "vmovups (%0), %%ymm2 \n\t"
                "vaddps %%ymm2, %%ymm0, %%ymm0 \n\t"
                "vmaxps %%ymm2, %%ymm5, %%ymm5 \n\t"
                "vminps %%ymm2, %%ymm6, %%ymm6 \n\t"
                "vmulps %%ymm2, %%ymm2, %%ymm3 \n\t"
                "vaddps %%ymm3, %%ymm1, %%ymm1 \n\t"
                :: "r"(ptr_w) :"memory");
            ptr_w += 8;
        }

        __asm__ __volatile__(
            "vdpps $0xF1, %%ymm0, %%ymm4, %%ymm2 \n\t"
            "vdpps $0xF2, %%ymm1, %%ymm4, %%ymm3 \n\t"
            "vaddps %%ymm2, %%ymm3, %%ymm3 \n\t"
            "vmovups %%ymm3, (%0) \n\t"
            "vmovups %%ymm5, (%1) \n\t"
            "vmovups %%ymm6, (%2) \n\t"
            :: "r"(tmp_arr),"r"(tmp_max),"r"(tmp_min):"memory");

        avg = tmp_arr[0]+tmp_arr[4];
        std = tmp_arr[1]+tmp_arr[5] - avg*avg;
        std = sqrtf(fmaxf(std, 0.0f));

        min = tmp_min[0];
        max = tmp_max[0];
        for(i=1;i<8;i++) {
            min  = fminf(tmp_min[i],min);
            max  = fmaxf(tmp_max[i],max);
        }

    }
    else {
        min = ptr[0];
        max = ptr[0];
        float M = ptr[0], S = 0.0f;
        for(i=1;i<length;i++) {
            float x = ptr[i];
            float old_M = M;
            M += (x - M) / (float)(i + 1);
            S += (x - old_M) * (x - M);
            min = fminf(x, min);
            max = fmaxf(x, max);
        }
        avg = M;
        std = sqrtf(S / numel);
    }
}

inline void zero_mean(float*ptr,const uint32 length, const float avg) {
    
    uint32 i;
    if( should_use_avx2(length) ) {

        float*ptr_w = (float*)ptr;
        __asm__ __volatile__(
            "vbroadcastss (%0), %%ymm0\n\t"
        :: "r"(&avg) :"memory");

        for(i=0;i<length;i+=8) {

            __asm__ __volatile__(
                "vmovups (%0), %%ymm2 \n\t"
                "vsubps %%ymm0, %%ymm2, %%ymm3 \n\t"
                "vmovups %%ymm3, (%0) \n\t"
            :: "r"(ptr_w) :"memory");

            ptr_w += 8;
        }

    }
    else {
        for(i=0;i<length;i++) {
            ptr[i] = (ptr[i]-avg);
        }
    }
}

inline bool normalize(float*ptr,const uint32 length, const float avg, const float old_std, const float new_std=1) {
    
    if(old_std < SUSAN_FLOAT_TOL || std::isnan(old_std) || std::isinf(old_std) )
        return false;

    float scale_factor = new_std/old_std;

    uint32 i;
    if( should_use_avx2(length) ) {

        float*ptr_w = (float*)ptr;
        __asm__ __volatile__(
            "vbroadcastss (%0), %%ymm0\n\t"
            "vbroadcastss (%1), %%ymm1\n\t"
        :: "r"(&avg), "r"(&scale_factor) :"memory");

        for(i=0;i<length;i+=8) {

            __asm__ __volatile__(
                "vmovups (%0), %%ymm2 \n\t"
                "vsubps %%ymm0, %%ymm2, %%ymm3 \n\t"
                "vmulps %%ymm1, %%ymm3, %%ymm4 \n\t"
                "vmovups %%ymm4, (%0) \n\t"
            :: "r"(ptr_w) :"memory");

            ptr_w += 8;
        }

    }
    else {
        for(i=0;i<length;i++) {
            ptr[i] = (ptr[i]-avg)*scale_factor;
        }
    }
    
    return true;
}

inline bool normalize_non_zero(float*ptr,const uint32 length) {
    
    int   num=0;
    float avg=0;
    float std=0;
    uint32 i;
    
    for(i=0;i<length;i++) {
        if( ptr[i] != 0 ) {
            avg+=ptr[i];
            num++;
        }
    }

    if( num == 0 )
        return false;

    avg = avg/num;

    for(i=0;i<length;i++) {
        if( ptr[i] != 0 ) {
            ptr[i] = ptr[i] - avg;
            std += (ptr[i]*ptr[i]);
        }
    }

    std = sqrtf( std/(num) );

    if(std < SUSAN_FLOAT_TOL )
    return false;


    for(i=0;i<length;i++) {
        if( ptr[i] != 0 ) {
            ptr[i] = ptr[i]/std;
        }
    }
    
    return true;
}

inline bool normalize_masked(float*ptr,const float*msk,const uint32 length, const float new_std=1) {

    float count = 0;
    float avg = 0;
    float std = 0;

    for(uint32 i=0; i<length; i++) {
        if( msk[i] > 0 ) {
            count += msk[i];
            avg += (ptr[i]*msk[i]);
        }
    }

    if( count < 2.0f )
        return false;

    avg = avg/count;

    for(uint32 i=0; i<length; i++) {
        if( msk[i] > 0 ) {
            float tmp = (ptr[i]-avg)*msk[i];
            std += tmp*tmp;
        }
    }

    std = sqrtf( std/(count-1) );

    for(uint32 i=0; i<length; i++) {
        float tmp = (ptr[i]-avg)*msk[i];
        ptr[i] = tmp*new_std/std;
    }

    return true;
}

inline void anscombe_transform(float*ptr,const uint32 length) {
    /// Applies the Anscombe transform (Poisson -> Normal)
    /// https:/doi.org/10.1093/biomet/35.3-4.246
    /// TODO: Speed this up!
    uint32 i;
    float avg = 0;
    for(i=0;i<length;i++) {
        float x = ptr[i] + (3.0f/8.0f);
        ptr[i]  = 2*sqrtf( fmaxf(x,0.0f) );
        avg    += ptr[i];
    }

    avg = avg/((float)length);

    for(i=0;i<length;i++) {
        ptr[i] = ptr[i] - avg;
    }
}

inline void generalized_anscombe_transform_zero_mean(float*ptr,const uint32 length) {
    /// Modified from: https:/doi.org/10.1109/TIP.2012.2202675
    /// TODO: Speed this up!
    //vv = vv + (3.0/8.0) + 1
    //vv = 2*np.sqrt( np.maximum(vv,0.0) )

    uint32 i;
    float avg = 0;
    for(i=0;i<length;i++) {
        float x = ptr[i] + (3.0f/8.0f) + 1.0f;
        ptr[i]  = 2*sqrtf( fmaxf(x,0.0f) );
        avg    += ptr[i];
    }

    avg = avg/((float)length);

    for(i=0;i<length;i++) {
        ptr[i] = ptr[i] - avg;
    }
}

inline void vst(float*ptr,const uint32 length,const float std=1.0) {
    /// TODO: Speed this up!
    float min_val = ptr[0];
    for(int i=0;i<length;i++)
        min_val = fminf(min_val,ptr[i]);

    float mean=0;
    for(int i=0;i<length;i++) {
        ptr[i] = 2*sqrtf( (ptr[i]-min_val)/std + (3.0f/8.0f) );
        mean += ptr[i];
    }

    mean = mean/length;
    for(int i=0;i<length;i++)
        ptr[i] = ptr[i] - mean;
}

inline void enforce_hermitian(double*p_vol, double*p_wgt, const uint32 M, const uint32 N) {
    uint32 N_h = N/2;
    for(int z=1; z<N; z++) {
        int y_start = ( z < N_h ) ? N_h : N_h + 1;
        for(int y=y_start; y<N; y++ ) {
            int ix_a = y*M + z*M*N;
            int ix_b = (N-y)*M + (N-z)*M*N;

            double vol_real = p_vol[2*ix_a  ] + p_vol[2*ix_b  ];
            double vol_imag = p_vol[2*ix_a+1] - p_vol[2*ix_b+1];
            p_vol[2*ix_a  ] = vol_real;
            p_vol[2*ix_a+1] = vol_imag;
            p_vol[2*ix_b  ] = vol_real;
            p_vol[2*ix_b+1] =-vol_imag;

            double wgt  = p_wgt[ix_a] + p_wgt[ix_b];
            p_wgt[ix_a] = wgt;
            p_wgt[ix_b] = wgt;
        }
    }
}

inline void invert(double*ptr, const uint32 length) {
    uint32 i;
    for(i=0;i<length;i++) {
        if( fabs(ptr[i]) < SUSAN_FLOAT_TOL )
            ptr[i] = 1;
        else {
            ptr[i] = 1/ptr[i];
        }
    }
}

inline void radial_avg(float*r_avg, float*r_wgt, const int L, const float*p_in, const int N, const single r_min, const single r_max) {
    int center = N/2;
    int range = (int)ceil(r_max)+1;
    range = fminf(range,N/2);

    memset(r_avg,0,sizeof(float)*L);
    memset(r_wgt,0,sizeof(float)*L);

    for(int j=center-range; j<=center+range; j++) {
        int j_off = j*N;
        float y = j - center;
        for(int i=center-range; i<=center+range; i++) {
            float x = i - center;
            float r = sqrtf(x*x+y*y);
            if( r >= r_min && r <= r_max ) {
                float val = p_in[ i + j_off ];
                int idx = (int)roundf(r);
                if( idx < L ) {
                    r_avg[idx] += val;
                    r_wgt[idx] += 1;
                }
            }
        }
    }

    for(int l=0; l<L; l++) {
        if( r_wgt[l] > SUSAN_FLOAT_TOL ) {
            r_avg[l] = r_avg[l]/r_wgt[l];
        }
        else
            r_avg[l] = 0;
    }
}

inline void radial_avg(float*r_avg, float*r_wgt, const int L, const float*p_in, const int N) {
    radial_avg(r_avg,r_wgt,L,p_in,N,0,N/2);
}

inline void fftshift_1D_batch(float*p_data,const int N,const int K) {
    float a,b;
    int Nh = N/2;
    for(int k=0;k<K;k++) {
        float*p_work = p_data + k*N;
        for(int i=0;i<Nh;i++) {
            a = p_work[i   ];
            b = p_work[i+Nh];
            p_work[i   ] = b;
            p_work[i+Nh] = a;
        }
    }
}

inline void expand_ps_hermitian(float*p_herm, const double*p_ps, const float scale, const uint32 M, const uint32 N, const uint32 K) {
    for(int k=0;k<K;k++) {
        const double* p_in  = p_ps+k*M*N;
        float* p_out = p_herm+k*N*N;
        for(int j=0;j<N;j++) {
            int y_in = j*M;
            int y_out_a = j*N;
            int y_out_b = (j>0) ? (N-j)*N : 0;
            for(int i=0;i<M;i++) {
                int x_out_a = (i<(N/2)) ? i+N/2 : 0;
                int x_out_b = (N/2)-i;
                double val = p_in[i+y_in]/scale;
                p_out[x_out_a + y_out_a] = val;
                p_out[x_out_b + y_out_b] = val;
            }
        }
    }
}

inline void randn(float*ptr,const uint32 length,const float u=0, const float s=1) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> d{u,s};

    for(uint32 i=0;i<length;i++) {
        ptr[i] = d(gen);
    }
}

inline void rand(float*ptr,const uint32 length,const float scale=1, const float offset=0) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<float> d(0.0,1.0);

    for(uint32 i=0;i<length;i++) {
        ptr[i] = scale*d(gen)+offset;
    }
}

inline void denoise_l0(float*p_out,const float*p_in,const uint32 length,const float lambda,const float rho=1.0f) {
    for(int i=0;i<length;i++) {
        float thresh = (fabsf(p_in[i]) > lambda) ? p_in[i] : 0.0f;
        p_out[i] = rho*thresh + (1.0f-rho)*p_in[i];
    }
}

class Timing {
protected:
    struct timeval starting_time;

public:
    Timing() {
        tic();
    }

    void tic() {
        gettimeofday(&starting_time, NULL);
    }

    uint32 toc() {
        struct timeval current_time;
        gettimeofday(&current_time, NULL);
        return current_time.tv_sec - starting_time.tv_sec;
    }

    void get_etc(int&days,int&hours,int&mins,int&secs,const int processed,const int total) {
        if( processed > 0 ) {
            if( total == processed ) {
                days =0;
                hours=0;
                mins =0;
                secs =0;
            }
            else {
                float scale = (float)(total-processed)/(float)(processed);
                float total_secs = scale*(float)toc();
                days = (int)floor(total_secs/(86400.0));
                total_secs = total_secs - 86400.0*days;
                hours = (int)floor(total_secs/(3600.0));
                total_secs = total_secs - 3600.0*hours;
                mins = (int)floor(total_secs/(60.0));
                secs = (int)ceilf(total_secs) - 60*mins;
                if(secs > 59)
                    secs = 59;
            }
        }
        else {
            days =99;
            hours=23;
            mins =59;
            secs =59;
        }
    }

};

}

#endif /// MATH_CPU_H




