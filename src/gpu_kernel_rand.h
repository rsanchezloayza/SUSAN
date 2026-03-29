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

#ifndef GPU_KERNEL_RAND_H
#define GPU_KERNEL_RAND_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "curand_kernel.h"

#include "gpu.h"
#include "gpu_kernel.h"

#define SUSAN_RAND_MAX (0x0800)

using namespace GpuKernels;

namespace GpuKernelsRand {

__global__ void rand_setup(curandState *state, unsigned long long seed,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        long idx = get_3d_idx(ss_idx,ss_siz);
        curand_init(seed+idx, 0, 0, state+idx);
    }
}

__global__ void gen_normal(float*p_out,curandState *state,const float avg, const float std,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float tmp = curand_normal(state+idx);
        p_out[idx] = std*tmp + avg;
    }
}

__global__ void gen_normal(float*p_out,curandState *state,const float2*p_avg_std,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);
        float tmp = curand_normal(state+idx);
        float2 avg_std = p_avg_std[ss_idx.z];
        p_out[idx] = avg_std.y*tmp + avg_std.x;
    }
}

__global__ void gen_normal_box_muller(float*p_out,uint32*state,const uint32 new_offset,const float2*p_avg_std,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long idx = get_3d_idx(ss_idx,ss_siz);

        bool parity_flag = (ss_idx.x&0x01) < 1;

        uint32 mask   = SUSAN_RAND_MAX-1;
        float  scale  = ((float)SUSAN_RAND_MAX);
        float  offset = 1.0/(2.0*(float)SUSAN_RAND_MAX);

        float X,Y,out;
        X = (float)(state[idx  ]&mask);
        if( parity_flag && ( ss_idx.x < ss_siz.x-1 ) ) {
            Y = (float)(state[idx+1]&mask);
        }
        else {
            Y = (float)(state[idx-1]&mask);
        }

        X = (X/scale) + offset;
        Y = (Y/scale) + offset;

        float R = sqrt( -2*logf(X) );
        float A = 2*M_PI*Y;

        if( parity_flag )
            out = R*cos(A);
        else
            out = R*sin(A);

        float2 avg_std = p_avg_std[ss_idx.z];

        p_out[idx] = avg_std.y*out + avg_std.x;

        state[idx] = (state[idx]&mask) + (new_offset&mask);
    }
}



}

#endif /// GPU_KERNEL_RAND_H

