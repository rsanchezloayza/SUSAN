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

#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cufft.h"

#include "gpu.h"

namespace GpuKernels {

/// DEVICE FUNCTIONS:
    
#if __CUDA_ARCH__ < 600
__device__ double atomic_Add(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#else
__device__ double atomic_Add(double* address, double val) {
    return atomicAdd(address,val);
}
#endif

__device__ void SN2(float&a,float&b,float&tmp) {
    tmp = a;
    a = max(b,a);
    b = min(b,tmp);
}

__device__ int3 get_th_idx() {
    return make_int3(threadIdx.x + blockIdx.x*blockDim.x,threadIdx.y + blockIdx.y*blockDim.y,threadIdx.z + blockIdx.z*blockDim.z);
}

__device__ bool first_thread_in_block() {
    return (threadIdx.x==0) && (threadIdx.y==0) && (threadIdx.z==0);
}

__device__ __forceinline__ float hanning_1d(const int n,const int N){
    float f = float(n)/float(N-1);
    if (N <= 1)
        return 1.0f;
    else
        return 0.5f * (1-cosf(2.0f*M_PI*f));
}

__device__ __forceinline__ long get_2d_idx(const int x,const int y,const int3&ss_siz) {
    return x + y*ss_siz.x;
}

__device__ __forceinline__ long get_2d_idx(const int3&ss_idx,const int3&ss_siz) {
    return get_2d_idx(ss_idx.x,ss_idx.y,ss_siz);
}

__device__ __forceinline__ long get_3d_idx(const int x,const int y,const int z,const int X,const int Y) {
    return x + y*X + z*X*Y;
}

__device__ __forceinline__ long get_3d_idx(const int x,const int y,const int z,const int3&ss_siz) {
    return get_3d_idx(x,y,z,ss_siz.x,ss_siz.y);
}

__device__ __forceinline__ long get_3d_idx(const int3&ss_idx,const int3&ss_siz) {
    return get_3d_idx(ss_idx.x,ss_idx.y,ss_idx.z,ss_siz.x,ss_siz.y);
}

__device__ __forceinline__ long get_3d_idx(const uint3&ss_idx,const dim3&ss_siz) {
    return get_3d_idx(ss_idx.x,ss_idx.y,ss_idx.z,ss_siz.x,ss_siz.y);
}

__device__  __forceinline__ float get_kaisser_bessel_kernel_polyfit(const float t) {
    float t2 = t*t;
    float t4 = t2*t2;
    float t6 = t4*t2;
    float t8 = t6*t2;
    float rslt = 0.99939224;
    rslt -= 3.37246839*t2;
    rslt += 4.70532537*t4;
    rslt -= 3.26100335*t6;
    rslt += 0.93508816*t8;
    return rslt;
}

__device__ __forceinline__ float get_kaisser_bessel_correction_polyfit(const float t) {
    float t2 = t*t;
    float t4 = t2*t2;
    float t6 = t4*t2;
    float rslt = 1.10650522;
    rslt -= 0.51809905*t2;
    rslt += 0.40407865*t4;
    rslt -= 0.09941299*t6;
    return rslt;
}

__device__ __forceinline__ float sinc(const float t) {
    if( fabsf(t)<1e-6f ) return 1.0f;
    float pt = M_PI * t;
    return sinf(pt) / pt;

}

__device__ __forceinline__ int fftshift_idx(const int idx, const int center) {
    return (idx<center) ? idx + center : idx - center;
}

__device__ __forceinline__ float l2_distance(const float x, const float y) {
    return sqrtf( x*x + y*y );
}

__device__ __forceinline__ float l2_distance(const float x, const float y, const float z) {
    return sqrtf( x*x + y*y + z*z );
}

__device__ __forceinline__ void insert_into_stk(double*p_acc, double*p_wgt, float val, const float x, const float y, const int z, const int3 ss_siz) {
    int x0 = int(floorf(x));
    int y0 = int(floorf(y));

    float dx = x - x0;
    float dy = y - y0;

    if(x0>=0 && y0>=0 && (x0+1)<ss_siz.x && (y0+1)<ss_siz.y) {
        float w00 = (1.0f-dx)*(1.0f-dy);
        float w10 = (     dx)*(1.0f-dy);
        float w01 = (1.0f-dx)*(     dy);
        float w11 = dx*dy;

        long int idx00 = get_3d_idx(x0  ,y0  ,z,ss_siz);
        long int idx10 = get_3d_idx(x0+1,y0  ,z,ss_siz);
        long int idx01 = get_3d_idx(x0  ,y0+1,z,ss_siz);
        long int idx11 = get_3d_idx(x0+1,y0+1,z,ss_siz);

        atomic_Add(p_acc+idx00,w00*val);
        atomic_Add(p_wgt+idx00,w00    );

        atomic_Add(p_acc+idx10,w10*val);
        atomic_Add(p_wgt+idx10,w10    );

        atomic_Add(p_acc+idx01,w01*val);
        atomic_Add(p_wgt+idx01,w01    );

        atomic_Add(p_acc+idx11,w11*val);
        atomic_Add(p_wgt+idx11,w11    );
    }
}

__device__ __forceinline__ void insert_into_stk(float*p_acc, float*p_wgt, float val, const float x, const float y, const int z, const int3 ss_siz) {
    int x0 = int(floorf(x));
    int y0 = int(floorf(y));

    float dx = x - x0;
    float dy = y - y0;

    if(x0>=0 && y0>=0 && (x0+1)<ss_siz.x && (y0+1)<ss_siz.y) {
        float w00 = (1.0f-dx)*(1.0f-dy);
        float w10 = (     dx)*(1.0f-dy);
        float w01 = (1.0f-dx)*(     dy);
        float w11 = dx*dy;

        long int idx00 = get_3d_idx(x0  ,y0  ,z,ss_siz);
        long int idx10 = get_3d_idx(x0+1,y0  ,z,ss_siz);
        long int idx01 = get_3d_idx(x0  ,y0+1,z,ss_siz);
        long int idx11 = get_3d_idx(x0+1,y0+1,z,ss_siz);

        atomicAdd(p_acc+idx00,w00*val);
        atomicAdd(p_wgt+idx00,w00    );

        atomicAdd(p_acc+idx10,w10*val);
        atomicAdd(p_wgt+idx10,w10    );

        atomicAdd(p_acc+idx01,w01*val);
        atomicAdd(p_wgt+idx01,w01    );

        atomicAdd(p_acc+idx11,w11*val);
        atomicAdd(p_wgt+idx11,w11    );
    }
}

__device__ __forceinline__ float extract_from_stk(const float* p_data, const float x, const float y, const int z, const int3 ss_siz) {
    int x0 = int(floorf(x));
    int y0 = int(floorf(y));

    float dx = x - x0;
    float dy = y - y0;

    if (x0 < 0 || y0 < 0 || (x0 + 1) >= ss_siz.x || (y0 + 1) >= ss_siz.y)
        return 0.0f;

    float w00 = (1.0f-dx)*(1.0f-dy);
    float w10 = (     dx)*(1.0f-dy);
    float w01 = (1.0f-dx)*(     dy);
    float w11 = dx*dy;

    long int idx00 = get_3d_idx(x0  ,y0  ,z,ss_siz);
    long int idx10 = get_3d_idx(x0+1,y0  ,z,ss_siz);
    long int idx01 = get_3d_idx(x0  ,y0+1,z,ss_siz);
    long int idx11 = get_3d_idx(x0+1,y0+1,z,ss_siz);

    float acc = 0;
    float wgt = 0;

    acc += w00*p_data[idx00];
    acc += w10*p_data[idx10];
    acc += w01*p_data[idx01];
    acc += w11*p_data[idx11];

    wgt += w00;
    wgt += w10;
    wgt += w01;
    wgt += w11;

    return (wgt > 0.0f) ? acc/wgt : 0.0f;
}

__device__ void get_xyR(float&x,float&y,float&R,const float in_x,const float in_y) {
    x = in_x;
    y = in_y;
    R = l2_distance(x,y);
}

__device__ void get_xyR_unit(float&x,float&y,float&R,const float in_x,const float in_y) {
    x = in_x;
    y = in_y;
    R = l2_distance(x,y);
    if( R<0.5 ) R = 1;
    x = x/R;
    y = y/R;
}

__device__ float get_bp_wgt(const float min_R,const float max_R,const float rolloff,const float R) {
    float roll_w = max(rolloff,1.0);
    float ang,wgt_lp,wgt_hp;
    ang = (R-max_R)/roll_w;
    ang = M_PI*min(max(ang,0.0),1.0);
    wgt_lp = 0.5*cosf(ang) + 0.5;

    ang = (min_R-R)/roll_w;
    ang = M_PI*min(max(ang,0.0),1.0);
    wgt_hp = 0.5*cosf(ang) + 0.5;

    return wgt_lp*wgt_hp;
}

/// HOST FUNCTIONS:
__global__ void rotate_180_stk(float*p_out,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long ix_in  = get_3d_idx(ss_idx,ss_siz);
        ss_idx.x    = ss_siz.x - ss_idx.x;
        ss_idx.y    = ss_siz.y - ss_idx.y;
        long ix_out = get_3d_idx(ss_idx,ss_siz);

        float val = p_in[ix_in];
        p_out[ix_out] = val;
    }
}

__global__ void fftshift2D(float*p_work,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    int Xh = ss_siz.x/2;
    int Yh = ss_siz.y/2;

    if( ss_idx.x < Xh && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long ix_A = get_3d_idx(ss_idx,ss_siz);
        ss_idx.x = ss_idx.x + Xh;
        ss_idx.y = fftshift_idx(ss_idx.y,Yh);
        long ix_B = get_3d_idx(ss_idx,ss_siz);

        float in_A = p_work[ix_A];
        float in_B = p_work[ix_B];

        p_work[ix_A] = in_B;
        p_work[ix_B] = in_A;
    }
}

__global__ void fftshift2D(float2*p_work,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    int Yh = ss_siz.y/2;

    if( ss_idx.x < ss_siz.x && ss_idx.y < Yh && ss_idx.z < ss_siz.z ) {

        long ix_A = get_3d_idx(ss_idx,ss_siz);
        ss_idx.y = fftshift_idx(ss_idx.y,Yh);
        long ix_B = get_3d_idx(ss_idx,ss_siz);

        float2 in_A = p_work[ix_A];
        float2 in_B = p_work[ix_B];

        p_work[ix_A] = in_B;
        p_work[ix_B] = in_A;
    }
}

__global__ void fftshift3D(float*p_work, const int N) {

    int3 ss_idx = get_th_idx();

    int center = N/2;

    if( ss_idx.x < center && ss_idx.y < N && ss_idx.z < N ) {

        int3 ss_siz = make_int3(N,N,N);
        long ix_A = get_3d_idx(ss_idx,ss_siz);
        ss_idx.x = ss_idx.x + center;
        ss_idx.y = fftshift_idx(ss_idx.y,center);
        ss_idx.z = fftshift_idx(ss_idx.z,center);
        long ix_B = get_3d_idx(ss_idx,ss_siz);

        float val_A = p_work[ ix_A ];
        float val_B = p_work[ ix_B ];

        p_work[ ix_A ] = val_B;
        p_work[ ix_B ] = val_A;

    }
}

__global__ void fftshift3D(float2*p_work, const int M,const int N) {

    int3 ss_idx = get_th_idx();

    int center = N/2;

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < center ) {

        int3 ss_siz = make_int3(M,N,N);
        long ix_A = get_3d_idx(ss_idx,ss_siz);
        ss_idx.y = fftshift_idx(ss_idx.y,center);
        ss_idx.z = fftshift_idx(ss_idx.z,center);
        long ix_B = get_3d_idx(ss_idx,ss_siz);

        float2 val_A = p_work[ ix_A ];
        float2 val_B = p_work[ ix_B ];

        p_work[ ix_A ] = val_B;
        p_work[ ix_B ] = val_A;

    }
}

__global__ void fftshift3D(double*p_work, const int M,const int N) {

    int3 ss_idx = get_th_idx();

    int center = N/2;

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < center ) {

        int3 ss_siz = make_int3(M,N,N);
        long ix_A = get_3d_idx(ss_idx,ss_siz);
        ss_idx.y = fftshift_idx(ss_idx.y,center);
        ss_idx.z = fftshift_idx(ss_idx.z,center);
        long ix_B = get_3d_idx(ss_idx,ss_siz);

        double val_A = p_work[ ix_A ];
        double val_B = p_work[ ix_B ];

        p_work[ ix_A ] = val_B;
        p_work[ ix_B ] = val_A;

    }
}

/// Use right after fft3.exec() or just before ifft3.exec()
__global__ void sampling_correction_3D(float2*p_work, const float correction, const int M,const int N) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < N ) {

        if( (ss_idx.x == 0 || ss_idx.x == N/2) &&
            (ss_idx.y == 0 || ss_idx.y == N/2) &&
            (ss_idx.z == 0 || ss_idx.z == N/2))
        {
            int3 siz = make_int3(M,N,N);
            long idx = get_3d_idx(ss_idx,siz);
            float2 val = p_work[ idx ];
            val.x *= correction;
            val.y *= correction;
            p_work[ idx ] = val;
        }
    }
}

/// Use right after batch fft2.exec() or just before batch ifft2.exec()
__global__ void sampling_correction_2D(float2*p_work,const float correction,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        if( (ss_idx.x == 0 || ss_idx.x == ss_siz.y/2) &&
            (ss_idx.y == 0 || ss_idx.y == ss_siz.y/2))
        {
            long idx = get_3d_idx(ss_idx,ss_siz);
            float2 val = p_work[ idx ];
            val.x *= correction;
            val.y *= correction;
            p_work[ idx ] = val;
        }
    }
}

__global__ void real_to_complex(float2*p_out,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = {0,0};
        val.x = p_in[idx];
        p_out[idx] = val;
    }
}

__global__ void analytical_signal_1D(float2*p_work,const int M,const int N,const int R,const int K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < N && ss_idx.y < (R*K) && ss_idx.z == 0 ) {

        long int idx = ss_idx.x + ss_idx.y*N;
        float2 val = p_work[idx];

        if( ss_idx.x>0 && ss_idx.x<(N/2) ) {
            val.x *= 2.0f;
            val.y *= 2.0f;
        }
        else if( ss_idx.x>(N/2)) {
            val.x = 0.0f;
            val.y = 0.0f;
        }

        p_work[idx] = val;
    }
}

__global__ void as_amplitude_normalize_clamp(float*p_out,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_in[idx];
        float den = cuCabsf(val);
        if(den<1e-10)
            den = 1.0;
        float rslt = ((val.x/den)/2.0f) + 0.5f;
        p_out[idx] = fminf(fmaxf(rslt,0.0f),1.0f);
    }
}

__global__ void load_surf(cudaSurfaceObject_t out_surf,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_dilate_1(cudaSurfaceObject_t out_surf,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v = p_in[ get_3d_idx(ss_idx,ss_siz) ];

        if( (ss_idx.x > 0) && (ss_idx.x < (ss_siz.x-1)) ) {
            if( (ss_idx.y > 0) && (ss_idx.y < (ss_siz.y-1)) ) {

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y-1,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y  ,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y+1,ss_idx.z,ss_siz) ]);
            }
        }

        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_dilate_2(cudaSurfaceObject_t out_surf,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v = p_in[ get_3d_idx(ss_idx,ss_siz) ];

        if( (ss_idx.x > 1) && (ss_idx.x < (ss_siz.x-2)) ) {
            if( (ss_idx.y > 1) && (ss_idx.y < (ss_siz.y-2)) ) {

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y-2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y-2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y-2,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y-1,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y  ,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y+1,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y+2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y+2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y+2,ss_idx.z,ss_siz) ]);
            }
        }

        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_dilate_3(cudaSurfaceObject_t out_surf,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v = p_in[ get_3d_idx(ss_idx,ss_siz) ];

        if( (ss_idx.x > 2) && (ss_idx.x < (ss_siz.x-3)) ) {
            if( (ss_idx.y > 2) && (ss_idx.y < (ss_siz.y-3)) ) {

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y-3,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y-3,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y-3,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y-2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y-2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y-2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y-2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y-2,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-3,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y-1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+3,ss_idx.y-1,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-3,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y  ,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+3,ss_idx.y  ,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-3,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y+1,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+3,ss_idx.y+1,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-2,ss_idx.y+2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y+2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y+2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y+2,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+2,ss_idx.y+2,ss_idx.z,ss_siz) ]);

                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x-1,ss_idx.y+3,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x  ,ss_idx.y+3,ss_idx.z,ss_siz) ]);
                v = fmaxf(v,p_in[ get_3d_idx(ss_idx.x+1,ss_idx.y+3,ss_idx.z,ss_siz) ]);
            }
        }

        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_3(cudaSurfaceObject_t out_surf,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        surf3Dwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_3(cudaSurfaceObject_t out_surf,const float2*p_in,const int3 ss_siz,bool conjugate=false) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 v = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        if( conjugate )
            v.y = -v.y;
        surf3Dwrite<float2>(v,out_surf,ss_idx.x*sizeof(float2), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_abs(cudaSurfaceObject_t out_surf,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float v = cuCabsf(val);
        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_real(cudaSurfaceObject_t out_surf,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        surf2DLayeredwrite<float>(val.x,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void load_surf_real_positive(cudaSurfaceObject_t out_surf,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float v = max(val.x,0.0f);
        surf2DLayeredwrite<float>(v,out_surf,ss_idx.x*sizeof(float), ss_idx.y, ss_idx.z);
    }
}

__global__ void conv_gauss_1D_X(float*p_out,const float*p_in,const float num,const float scl,const int3 ss_siz,const bool is_hermitian) {
    int3 ss_idx = get_th_idx();

    if( ss_idx.x >= ss_siz.x || ss_idx.y >= ss_siz.y || ss_idx.z >= ss_siz.z )
        return;

    float val = 0.0f;

    for(int x=-15;x<=15;x++) {
        float wgt = expf(-num*float(x)*float(x))/scl;

        int X = ss_idx.x + x;
        int Y = ss_idx.y;

        if (X < 0) {
            X = -X;
            if (is_hermitian) {
                Y = ss_siz.y - Y;
                if (Y == ss_siz.y) Y = 0;
            }
        }
        if (X >= ss_siz.x)
            X = 2*(ss_siz.x-1)-X;

        val += wgt*p_in[ get_3d_idx(X,Y,ss_idx.z,ss_siz) ];
    }
    p_out[get_3d_idx(ss_idx,ss_siz)] = val;
}

__global__ void conv_gauss_1D_Y(float*p_out,const float*p_in,const float num,const float scl,const int3 ss_siz) {
    int3 ss_idx = get_th_idx();

    if( ss_idx.x >= ss_siz.x || ss_idx.y >= ss_siz.y || ss_idx.z >= ss_siz.z )
        return;

    float val = 0.0f;

    for(int y=-15;y<=15;y++) {
        float wgt = expf(-num*float(y)*float(y))/scl;

        int Y = ss_idx.y + y;

        if (Y < 0)
            Y = -Y;
        if (Y >= ss_siz.y)
            Y = 2*(ss_siz.y-1)-Y;

        val += wgt*p_in[ get_3d_idx(ss_idx.x,Y,ss_idx.z,ss_siz) ];
    }
    p_out[get_3d_idx(ss_idx,ss_siz)] = val;
}

__global__ void conv_gaussian(float*p_out,const float*p_in,float num,float scl,const int3 ss_siz, const bool is_complex=true) {
    ///    num    scl:
    /// 0.5000, 6.2831
    /// 0.2500,12.5372
    /// 0.1250,23.9907

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float val = 0;

        for(int y=-4;y<5;y++) {
            for(int x=-4;x<5;x++) {
                float r2  = float(x*x + y*y);
                float wgt = expf(-num*r2)/scl;

                int X = ss_idx.x + x;
                int Y = ss_idx.y + y;

                if( X < 0 ) {
                    X = -X;
                    if( is_complex )
                        Y = ss_siz.y-Y;
                    if( Y == ss_siz.y )
                        Y = 0;
                }
                if( X >= ss_siz.x )
                    X = 2*(ss_siz.x-1)-X;

                if( Y < 0 )
                    Y = -Y;
                if( Y >= ss_siz.y )
                    Y = 2*(ss_siz.y-1)-Y;

                val += wgt*p_in[ get_3d_idx(X,Y,ss_idx.z,ss_siz) ];
            }
        }

        p_out[get_3d_idx(ss_idx,ss_siz)] = val;
    }
}

__global__ void stk_hanning(float*p_val,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        long idx = get_3d_idx(ss_idx,ss_siz);
        float wx = hanning_1d(ss_idx.x,ss_siz.x);
        float wy = hanning_1d(ss_idx.y,ss_siz.y);
        p_val[idx] *= (wx*wy);
    }
}

__global__ void arr_hanning(float*p_val,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        long idx = get_3d_idx(ss_idx,ss_siz);
        float wx = hanning_1d(ss_idx.x,ss_siz.x);
        p_val[idx] *= wx;
    }
}

__global__ void stk_medfilt_k3(float*p_out,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float v0,v1,v2,v3,v4,v5,v6,v7,v8,tmp;
        int y, x;

        /// unrolled to avoid using local arrays

        /// Y = -1
        y = ss_idx.y - 1;
        if( y < 0 ) y = -y;

        x = ss_idx.x - 1;
        if( x < 0 ) x = -x;
        v0 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        x = ss_idx.x;
        v1 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        x = ss_idx.x + 1;
        if( x >= ss_siz.x ) x = 2*(ss_siz.x-1)-x;
        v2 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        /// Y = 0
        y = ss_idx.y;

        x = ss_idx.x - 1;
        if( x < 0 ) x = -x;
        v3 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        x = ss_idx.x;
        v4 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        x = ss_idx.x + 1;
        if( x >= ss_siz.x ) x = 2*(ss_siz.x-1)-x;
        v5 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        /// Y = +1
        y = ss_idx.y + 1;
        if( y >= ss_siz.y ) y = 2*(ss_siz.y-1)-y;

        x = ss_idx.x - 1;
        if( x < 0 ) x = -x;
        v6 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        x = ss_idx.x;
        v7 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        x = ss_idx.x + 1;
        if( x >= ss_siz.x ) x = 2*(ss_siz.x-1)-x;
        v8 = p_in[ get_3d_idx(x,y,ss_idx.z,ss_siz) ];

        /// SORT!
        SN2(v1,v2,tmp); SN2(v4,v5,tmp); SN2(v7,v8,tmp);
        SN2(v0,v1,tmp); SN2(v3,v4,tmp); SN2(v6,v7,tmp);
        SN2(v1,v2,tmp); SN2(v4,v5,tmp); SN2(v7,v8,tmp);
        SN2(v0,v3,tmp); SN2(v5,v8,tmp); SN2(v4,v7,tmp);
        SN2(v3,v6,tmp); SN2(v1,v4,tmp); SN2(v2,v5,tmp);
        SN2(v4,v7,tmp); SN2(v4,v2,tmp); SN2(v6,v4,tmp);
        SN2(v4,v2,tmp);

        p_out[get_3d_idx(ss_idx,ss_siz)] = v4;
    }
}

__global__ void get_avg_std(float*p_std, float*p_avg, const float*p_in, const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float N = ss_siz.x*ss_siz.y;
        long idx = get_3d_idx(ss_idx,ss_siz);
        float val = p_in[idx];
        float avg = val/N;
        float std = val*avg;
        atomicAdd(p_avg+ss_idx.z,avg);
        atomicAdd(p_std+ss_idx.z,std);
    }
}

__global__ void get_std_from_fourier_stk(float*p_std,const float2*p_data,const float3 bandpass,const int3 ss_siz) {

    __shared__ float local_std[1];
    if( first_thread_in_block() )
        local_std[0] = 0;
    __syncthreads();

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float R = l2_distance(ss_idx.x,ss_idx.y - ss_siz.y/2);
        float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

        if( (bp > 0.05) && (R > 0.5) ) {
            long idx = get_3d_idx(ss_idx,ss_siz);
            float2 val = p_data[idx];
            val.x *= bp;
            val.y *= bp;
            float acc = cuCabsf(val);
            acc = acc*acc;
            if( ss_idx.x > 0 )
                acc *= 2;
            atomicAdd( local_std , acc );
        }
        __syncthreads();

        if(first_thread_in_block()) {
            float acc = local_std[0];
            atomicAdd( p_std+ss_idx.z , acc );
        }
    }
}

__global__ void apply_std_to_fourier_stk(float2*p_data,const float*p_std,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_data[idx];
        float wgt = p_std[ss_idx.z];
        wgt   = sqrtf(wgt);
        val.x = val.x/wgt;
        val.y = val.y/wgt;
        p_data[idx] = val;
    }
}

__global__ void zero_avg_one_std(float*p_in, const float*p_std, const float*p_avg, const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float avg = p_avg[ss_idx.z];
        float std = sqrtf( p_std[ss_idx.z] - p_avg[ss_idx.z]*p_avg[ss_idx.z] );

        long idx = get_3d_idx(ss_idx,ss_siz);
        float val = p_in[idx];
        p_in[idx] = (val-avg)/std;
    }
}

__global__ void expand_ps_hermitian(float*p_out,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int Nh = ss_siz.y/2;
        int x = ss_idx.x - Nh;
        int y = ss_idx.y - Nh;

        if( x<0 ) {
            x = -x;
            y = -y;
        }

        x = x+Nh;
        y = y+Nh;

        if(x<0) x = Nh+x;
        if(y<0) y = ss_siz.y+y;

        if(x>=Nh) x = x-Nh;
        if(y>=ss_siz.y) y = y-ss_siz.y;

        float val = p_in[ x + y*(Nh+1) + ss_idx.z*ss_siz.y*(Nh+1) ];

        p_out[get_3d_idx(ss_idx,ss_siz)] = val;
    }
}

__global__ void apply_circular_mask(float*p_out,const float*p_in,const float2*p_bp,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float val = 0;

        float R = l2_distance(ss_idx.x-ss_siz.x/2,ss_idx.y-ss_siz.y/2);

        float w = get_bp_wgt(p_bp[ss_idx.z].x,p_bp[ss_idx.z].y,10,R);

        if( w > 0.05 ) {
            val = w*p_in[ idx ];
        }
        p_out[idx] = val;
    }
}

__global__ void stk_scale(float*p_data,const float2*p_bp,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float val = p_data[idx];
        float w = p_bp[ss_idx.z].y;
        w = M_PI*w*w;
        val = p_data[ idx ]/w;
        p_data[idx] = val;
    }
}

__global__ void load_abs(float*p_out,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_in[idx];
        float v = cuCabsf(val);
        p_out[idx] = v;
    }
}

__global__ void load_ps(float*p_out,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_in[idx];
        float v = cuCabsf(val);
        p_out[idx] = (v*v);
    }
}

__global__ void radial_ps_avg(float*p_avg,float*p_wgt,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float x = ss_idx.x;
        float y = ss_idx.y-ss_siz.y/2;
        float R = l2_distance(x,y);
        int   r = (int)roundf(R);
        int idx = r + ss_siz.x*ss_idx.z;
        if( r < ss_siz.x ) {
            atomicAdd(p_avg + idx,val);
            atomicAdd(p_wgt + idx,1.0);
        }
    }
}

__global__ void radial_ps_avg_linearized(float*p_avg,float*p_wgt,const float*p_in,const float max_res,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int N  = ss_siz.y;
        int Nh = N/2;
        float x = ss_idx.x;
        float y = ss_idx.y - Nh;
        float s = l2_distance(x,y)/(Nh*max_res);

        if(s<1.0) {
            float r = s*s;
            float u = r*Nh;
            float val = p_in[get_3d_idx(ss_idx,ss_siz)];
            float w0 = 1.0f - (u - floorf(u));
            float w1 = u - floorf(u);

            #pragma unroll
            for(int b=0;b<2;b++) {
                int bin = (int)floorf(u) + b;
                float w = (b==0)? w0 : w1;

                #pragma unroll
                for(int sym=0;sym<2;sym++) {
                    int idx = (sym==0) ? (Nh+bin) : (Nh-bin);
                    if( idx>=0 && idx<N) {
                        atomicAdd(p_avg+idx+ss_idx.z*N, w*val);
                        atomicAdd(p_wgt+idx+ss_idx.z*N, w);
                    }
                }
            }
        }
    }
}

__global__ void radial_ps_avg(float*p_avg,float*p_wgt,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float  x = ss_idx.x;
        float  y = ss_idx.y-ss_siz.y/2;
        float  R = l2_distance(x,y);
        int    r = (int)roundf(R);
        int  idx = r + ss_siz.x*ss_idx.z;
        float out = cuCabsf(val);
        if( r < ss_siz.x ) {
            atomicAdd(p_avg + idx,out);
            atomicAdd(p_wgt + idx,1.0);
        }
    }
}

__global__ void radial_ps_avg_double_side(float*p_avg,float*p_wgt,const float*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float  x = ss_idx.x;
        float  y = ss_idx.y-ss_siz.y/2;
        float  R = l2_distance(x,y);
        int    r = (int)roundf(R);
        int  idx = r + ss_siz.y*ss_idx.z;
        if( r < ss_siz.x ) {
            atomicAdd(p_avg + idx,val);
            atomicAdd(p_wgt + idx,1.0);

            r = ss_siz.y - r;
            if( r < ss_siz.y ) {
                idx = r + ss_siz.y*ss_idx.z;
                atomicAdd(p_avg + idx,val);
                atomicAdd(p_wgt + idx,1.0);
            }
        }

    }
}

__global__ void radial_ps_norm(float2*p_data,const float*p_avg,const float*p_wgt,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = {0,0};
        float  x = ss_idx.x;
        float  y = ss_idx.y-ss_siz.y/2;
        float  R = l2_distance(x,y);
        int    r = (int)roundf(R);

        if( r < ss_siz.x ) {
            val = p_data[ get_3d_idx(ss_idx,ss_siz) ];
            int  idx = r + ss_siz.x*ss_idx.z;
            float avg = p_avg[idx];
            float wgt = p_wgt[idx];

            wgt = max(wgt,1.0);
            avg = avg/wgt;

            if( avg > 1e-7 ) {
                val = p_data[ get_3d_idx(ss_idx,ss_siz) ];
                val.x = val.x/avg;
                val.y = val.y/avg;
            }
            p_data[ get_3d_idx(ss_idx,ss_siz) ] = val;
        }
    }
}

__global__ void radial_frc_avg_stk(float*p_avg,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float    x = ss_idx.x;
        float    y = ss_idx.y-(ss_siz.y/2);
        float    R = l2_distance(x,y);
        int      r = (int)roundf(R);
        int    idx = r + ss_siz.x*ss_idx.z;
        float  out = (val.x*val.x) + (val.y*val.y);
        if( r < ss_siz.x ) {
            atomicAdd(p_avg + idx,out);
        }
    }
}

__global__ void radial_frc_avg_vol(float*p_avg,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = p_in[ get_3d_idx(ss_idx,ss_siz) ];
        float    x = ss_idx.x;
        float    y = ss_idx.y-(ss_siz.y/2);
        float    z = ss_idx.z-(ss_siz.z/2);
        float    R = l2_distance(x,y,z);
        int      r = (int)roundf(R);
        int    idx = r;
        float  out = (val.x*val.x) + (val.y*val.y);
        if( r < ss_siz.x ) {
            atomicAdd(p_avg + idx,out);
        }
    }
}

__global__ void radial_frc_acc(float*p_avg,const float ssnr_F,const float ssnr_S,const int K, const int M) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < K && ss_idx.y < 1 && ss_idx.z < 1 ) {

        float N = 2*(M-1);
        float*w_avg = p_avg + ss_idx.x*M;
        double avg;
        double ssnr  = 0.0;
        double scale = sqrt(N/2);

        w_avg[0] = 1.0*scale;
        for(int i=1;i<M;i++) {
            avg = sqrt(w_avg[i]);
            if(ssnr_S>1)
                ssnr = (1/(ssnr_S*exp(i*ssnr_F)));
            w_avg[i] = (avg + ssnr)*scale;
        }
    }
}

__global__ void radial_frc_norm_stk(float2*p_data,const float*p_avg,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = {0,0};
        float  x = ss_idx.x;
        float  y = ss_idx.y-ss_siz.y/2;
        float  R = l2_distance(x,y);
        int    r = (int)roundf(R);

        if( r < ss_siz.x ) {
            int    idx;
            double avg;

            idx  = r + ss_siz.x*ss_idx.z;
            avg  = p_avg[idx];

            if( avg > 1e-10 ) {
                val = p_data[ get_3d_idx(ss_idx,ss_siz) ];
                if( r > 0 ) {
                    val.x = val.x/avg;
                    val.y = val.y/avg;
                }
            }
        }

        p_data[ get_3d_idx(ss_idx,ss_siz) ] = val;
    }
}

__global__ void radial_frc_norm_vol(float2*p_data,const float*p_avg,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = {0,0};
        float    x = ss_idx.x;
        float    y = ss_idx.y-(ss_siz.y/2);
        float    z = ss_idx.z-(ss_siz.z/2);
        float    R = l2_distance(x,y,z);
        int      r = (int)roundf(R);

        if( (r < ss_siz.x) && (r > 0) ) {
            int    idx;
            double avg;

            idx  = r;
            avg  = p_avg[idx];

            if( avg > 1e-8 ) {
                val = p_data[ get_3d_idx(ss_idx,ss_siz) ];
                if( r > 0 ) {
                    val.x = val.x/avg;
                    val.y = val.y/avg;
                }
            }
        }

        p_data[ get_3d_idx(ss_idx,ss_siz) ] = val;
    }
}

__global__ void to_polar(float*p_out,const float*p_data,const float ang_step, const int3 ss_stk,const int3 ss_out) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_out.x && ss_idx.y < ss_out.y && ss_idx.z < ss_out.z ) {

        int   Nh  = ss_out.x/2;
        int   rad = ss_idx.x - Nh;
        float ang = ss_idx.y*ang_step;

        float x = rad*sinf(ang)+Nh;
        float y = rad*cosf(ang)+Nh;

        p_out[ get_3d_idx(ss_idx,ss_out) ] = extract_from_stk(p_data,x,y,ss_idx.z,ss_stk);
    }

}

__global__ void to_polar_complex(float2*p_out,const float*p_data,const float ang_step, const int3 ss_stk,const int3 ss_out) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_out.x && ss_idx.y < ss_out.y && ss_idx.z < ss_out.z ) {

        float2 val = {0,0};

        int   Nh  = ss_out.x/2;
        int   rad = ss_idx.x - Nh;
        float ang = ss_idx.y*ang_step;

        float x = rad*sinf(ang)+Nh;
        float y = rad*cosf(ang)+Nh;

        val.x = extract_from_stk(p_data,x,y,ss_idx.z,ss_stk);
        p_out[ get_3d_idx(ss_idx,ss_out) ] = val;
    }

}

__global__ void polar_to_cart_complex(float2*p_out,const float2*p_data,const float ang_step, const int N, const int R, const int K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < N && ss_idx.y < N && ss_idx.z < K ) {

        int   Nh  = N/2;

        float x = ss_idx.x - Nh;
        float y = ss_idx.y - Nh;

        float r = l2_distance(x,y);
        float a = atan2f(x,y);

        if(a<0)
            a += M_PI;

        if(a>=M_PI)
            a -= M_PI;

        r += Nh;
        a  = a / ang_step;

        float2 val = {0,0};

        if( (r>=0) && (r<(N-1)) ) {
            int r0 = floorf(r);
            int a0 = floorf(a);
            int r1 = r0 + 1;
            int a1 = a0 + 1;

            float wr = r - r0;
            float wa = a - a0;

            a0 = a0 % R;
            a1 = a1 % R;

            float2 v00 = p_data[ get_3d_idx(r0,a0,ss_idx.z,N,R) ];
            float2 v01 = p_data[ get_3d_idx(r0,a1,ss_idx.z,N,R) ];
            float2 v10 = p_data[ get_3d_idx(r1,a0,ss_idx.z,N,R) ];
            float2 v11 = p_data[ get_3d_idx(r1,a1,ss_idx.z,N,R) ];

            val.x = (1-wr)*(1-wa)*v00.x +
                    (1-wr)*(  wa)*v01.x +
                    (  wr)*(1-wa)*v10.x +
                    (  wr)*(  wa)*v11.x;
            val.y = (1-wr)*(1-wa)*v00.y +
                    (1-wr)*(  wa)*v01.y +
                    (  wr)*(1-wa)*v10.y +
                    (  wr)*(  wa)*v11.y;

        }

        p_out[ get_3d_idx(ss_idx.x,ss_idx.y,ss_idx.z,N,N) ] = val;
    }

}

__global__ void get_energy_stk(float*p_avg,const float*p_ctf,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float val = p_ctf[ get_3d_idx(ss_idx,ss_siz) ];
        int   idx = ss_siz.x*ss_idx.z;
        float out = val*val;
        atomicAdd(p_avg + idx,out);
    }
}

__global__ void divide_energy_stk(float*p_ctf,const float*p_avg,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int   idx = get_3d_idx(ss_idx,ss_siz);
        float val = p_ctf[idx];
        float wgt = p_avg[ss_siz.x*ss_idx.z];
        wgt = sqrtf( wgt/(ss_siz.x*ss_siz.y) );
        p_ctf[idx] = val/wgt;
    }
}

__global__ void divide(float*p_avg,const float*p_wgt,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx = get_3d_idx(ss_idx,ss_siz);
        float val = p_avg[ idx ];
        float wgt = p_wgt[ idx ];
        if( wgt < 1 ) wgt = 1;
        p_avg[ idx ] = val/wgt;
    }
}

__global__ void divide(cudaSurfaceObject_t surf,const float wgt,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        float2 val = surf3Dread<float2>(surf,ss_idx.x*sizeof(float2), ss_idx.y, ss_idx.z);
        val.x = val.x/wgt;
        val.y = val.y/wgt;
        surf3Dwrite<float2>(val,surf,ss_idx.x*sizeof(float2), ss_idx.y, ss_idx.z);
    }
}

__global__ void divide(float2*p_avg,const float wgt,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_avg[ idx ];
        val.x = val.x/wgt;
        val.y = val.y/wgt;
        p_avg[ idx ] = val;
    }
}

__global__ void divide(float*p_avg,const float wgt,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx = get_3d_idx(ss_idx,ss_siz);
        float val = p_avg[ idx ];
        val = val/wgt;
        p_avg[ idx ] = val;
    }
}

__global__ void substract(float*p_out,const float*p_sub,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx = get_3d_idx(ss_idx,ss_siz);
        float v1 = p_out[ idx ];
        float v2 = p_sub[ idx ];
        p_out[ idx ] = v1 - v2;
    }
}

__global__ void load_pad(float*p_out,const float*p_in,const int3 half_pad,const int3 ss_raw,const int3 ss_pad) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_raw.x && ss_idx.y < ss_raw.y && ss_idx.z < ss_raw.z ) {

        int idx_in  = get_3d_idx(ss_idx,ss_raw);
        int idx_out = get_3d_idx(ss_idx.x+half_pad.x,ss_idx.y+half_pad.y,ss_idx.z+half_pad.z,ss_pad);
        float data = p_in[idx_in];
        p_out[idx_out] = data;
    }

}

__global__ void load_pad(float2*p_out,const float2*p_in,const int3 half_pad,const int3 ss_raw,const int3 ss_pad) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_raw.x && ss_idx.y < ss_raw.y && ss_idx.z < ss_raw.z ) {

        int idx_in  = get_3d_idx(ss_idx,ss_raw);
        int idx_out = get_3d_idx(ss_idx.x+half_pad.x,ss_idx.y+half_pad.y,ss_idx.z+half_pad.z,ss_pad);
        float2 data = p_in[idx_in];
        p_out[idx_out] = data;
    }

}


__global__ void remove_pad_vol(float*p_out,const float*p_in,const int half_pad,const int3 ss_raw,const int3 ss_pad) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_raw.x && ss_idx.y < ss_raw.y && ss_idx.z < ss_raw.z ) {

        long idx_in  = get_3d_idx(ss_idx.x+half_pad,ss_idx.y+half_pad,ss_idx.z+half_pad,ss_pad);
        long idx_out = get_3d_idx(ss_idx,ss_raw);
        float data = p_in[idx_in];
        p_out[idx_out] = data;
    }

}

__global__ void remove_pad_stk(float*p_out,const float*p_in,const int half_pad,const int3 ss_raw,const int3 ss_pad) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_raw.x && ss_idx.y < ss_raw.y && ss_idx.z < ss_raw.z ) {

        long idx_in  = get_3d_idx(ss_idx.x+half_pad,ss_idx.y+half_pad,ss_idx.z,ss_pad);
        long idx_out = get_3d_idx(ss_idx,ss_raw);
        float data = p_in[idx_in];
        p_out[idx_out] = data;
    }

}

__global__ void subpixel_shift(float2*p_data,const Proj2D*g_ali,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx  = get_3d_idx(ss_idx,ss_siz);
        float2 data = p_data[idx];

        float x = ss_idx.x;
        float y = ss_idx.y-ss_siz.y/2;
        float N = ss_siz.y;

        float dx = g_ali[ss_idx.z].t.x;
        float dy = g_ali[ss_idx.z].t.y;

        float phase_shift = (x*dx + y*dy) / N;
        float sin_cos_arg = -2*M_PI*phase_shift;

        /// exp(-ix) = cos(x) - i sin(x);
        float tmp_real = data.x;
        float tmp_imag = data.y;
        float c = cosf(sin_cos_arg);
        float s = sinf(sin_cos_arg);
        data.x = tmp_real*c + tmp_imag*s;
        data.y = tmp_imag*c - tmp_real*s;

        p_data[idx] = data;
    }

}

__global__ void multiply(float2*p_out,const double2*p_acc,const double*p_wgt,const int3 ss_siz,const double scale=1.0) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        int idx = get_3d_idx(ss_idx,ss_siz);
        double2 val = p_acc[ idx ];
        double  wgt = p_wgt[ idx ];
        float2 rslt;
        rslt.x = (float)(val.x*wgt*scale);
        rslt.y = (float)(val.y*wgt*scale);
        p_out[ idx ] = rslt;
    }
}

__global__ void multiply(float2*p_out,const float2*p_in,const float*p_wgt,const int3 ss_siz,const double scale=1.0) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_in [ idx ];
        float  wgt = p_wgt[ idx ];
        float2 rslt;
        rslt.x = (float)(val.x*wgt*scale);
        rslt.y = (float)(val.y*wgt*scale);
        p_out[ idx ] = rslt;
    }
}

__global__ void multiply(float2*p_out,const float*p_wgt,const int3 ss_siz,const float scale=1.0) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_out[ idx ];
        float  wgt = p_wgt[ idx ];
        float2 rslt;
        rslt.x = (val.x*wgt*scale);
        rslt.y = (val.y*wgt*scale);
        p_out[ idx ] = rslt;
    }
}

__global__ void multiply(float2*p_out,const float2*p_in,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_out[ idx ];
        float2 wgt = p_in [ idx ];
        float2 rslt;
        rslt = cuCmulf(val,wgt);
        p_out[ idx ] = rslt;
    }
}

__global__ void print_proj2D(Proj2D*g_tlt,const int in_K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x == 0 && ss_idx.y == 0 && ss_idx.z == 0 ) {
        for( int z=0; z<in_K; z++ ) {
            //if( (z == 11) || (z == 21) || (z == 31) ) {
            if( z == 30 ) {
                printf("R[%d,0,:] = (%f,%f,%f)\n",z,g_tlt[z].R.xx,g_tlt[z].R.xy,g_tlt[z].R.xz);
                printf("R[%d,1,:] = (%f,%f,%f)\n",z,g_tlt[z].R.yx,g_tlt[z].R.yy,g_tlt[z].R.yz);
                printf("R[%d,2,:] = (%f,%f,%f)\n",z,g_tlt[z].R.zx,g_tlt[z].R.zy,g_tlt[z].R.zz);
            }
        }
    }
}

__global__ void rotate_post(Proj2D*g_tlt,Rot33 R,Proj2D*g_tlt_in,const int in_K) {

    /// R_out = R @ R_in
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < in_K*9 && ss_idx.y == 0 && ss_idx.z == 0 ) {

        int mod_z = ss_idx.x%9;
        int z = (ss_idx.x - mod_z)/9;
        int x = mod_z % 3;
        int y = (mod_z - x)/3;

        float *pR = &(R.xx);
        float *r_out = &(g_tlt[z].R.xx);
        float *r_in  = &(g_tlt_in[z].R.xx);

        float rslt = 0;

        /// rslt = pR * r_in
        /// rslt = R * g_tlt[z].R
        rslt += pR[y*3  ]*r_in[x  ];
        rslt += pR[y*3+1]*r_in[x+3];
        rslt += pR[y*3+2]*r_in[x+6];

        r_out[mod_z] = rslt;
    }

    __syncthreads();

    if( ss_idx.x < in_K && ss_idx.y == 0 && ss_idx.z == 0 ) {
        g_tlt[ss_idx.x].w = g_tlt_in[ss_idx.x].w;
    }

}

__global__ void rotate_pre(Proj2D*g_tlt,Rot33 R,Proj2D*g_tlt_in,const int in_K) {

    /// R_out = R_in @ R
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < in_K*9 && ss_idx.y == 0 && ss_idx.z == 0 ) {

        int mod_z = ss_idx.x%9;
        int z = (ss_idx.x - mod_z)/9;
        int x = mod_z % 3;
        int y = (mod_z - x)/3;

        float *pR = &(R.xx);
        float *r_out = &(g_tlt[z].R.xx);
        float *r_in  = &(g_tlt_in[z].R.xx);

        float rslt = 0;

        /// rslt = r_in * pR
        /// rslt = g_tlt[z].R * R
        rslt += r_in[y*3  ]*pR[x  ];
        rslt += r_in[y*3+1]*pR[x+3];
        rslt += r_in[y*3+2]*pR[x+6];

        r_out[mod_z] = rslt;
    }

    __syncthreads();

    if( ss_idx.x < in_K && ss_idx.y == 0 && ss_idx.z == 0 ) {
        g_tlt[ss_idx.x].w = g_tlt_in[ss_idx.x].w;
    }

}

__global__ void apply_radial_wgt(float2*p_data,const float w_total,float crowther_limit,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_data[ idx ];

        float w_off = 1/w_total;
        float w = ss_idx.x;
        w = fminf(w/crowther_limit,1.0);
        w = (1-w_off)*w + w_off;

        float K = fminf(crowther_limit,ss_siz.x);
        float w_inv = 1/w_total;

        float energy  = ( ss_siz.x - K );
        energy += ( (1-w_inv)*(1-w_inv) * ( (K-1)*K*(2*K-1) )/(6*crowther_limit*crowther_limit) );
        energy += ( 2*(1-w_inv)*w_inv*((K-1)*K) / (2*crowther_limit) );
        energy += ( (w_inv*w_inv)*K );
        energy  = sqrtf(energy);

        val.x = w*val.x/energy;
        val.y = w*val.y/energy;

        p_data[ idx ] = val;
    }
}

__global__ void apply_bandpass_fourier(float2*p_w,const float3 bandpass,const int M, const int N, const int K)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < K ) {

        long idx = ss_idx.x + M*ss_idx.y + M*N*ss_idx.z;

        float2 val = {0,0};

        float R = l2_distance(ss_idx.x,ss_idx.y - N/2);
        float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

        if( bp > 0.025 ) {
            val = p_w[ idx ];
            val.x *= bp;
            val.y *= bp;
        }

        p_w[ idx ] = val;
    }
}

__global__ void apply_bandpass_fourier(float2*p_w,const Defocus*p_def,const float3 bandpass,const int M, const int N, const int K)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < K ) {

        long idx = ss_idx.x + M*ss_idx.y + M*N*ss_idx.z;

        float2 val = {0,0};

        float max_R = bandpass.y;
        if( p_def[ss_idx.z].max_res > 0 )
            max_R = min(max_R,p_def[ss_idx.z].max_res);
        float R = l2_distance(ss_idx.x,ss_idx.y - N/2);
        float bp = get_bp_wgt(bandpass.x,max_R,bandpass.z,R);

        if( bp > 0.025 ) {
            val = p_w[ idx ];
            val.x *= bp;
            val.y *= bp;
        }

        p_w[ idx ] = val;
    }
}




__global__ void norm_complex(float2*p_data,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int idx = get_3d_idx(ss_idx,ss_siz);
        float2 val = p_data[ idx ];

        float wgt = l2_distance(val.x,val.y);
        if( wgt < 0.000001 )
            wgt = 1;

        val.x = val.x*wgt;
        val.y = val.y*wgt;

        p_data[ idx ] = val;
    }
}

}

#endif /// GPU_KERNEL_H


