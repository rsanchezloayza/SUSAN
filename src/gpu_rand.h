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

#ifndef GPU_RAND_H
#define GPU_RAND_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>

#include "datatypes.h"

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "curand_kernel.h"

#include "gpu.h"
#include "gpu_kernel.h"
#include "gpu_kernel_rand.h"

namespace GpuRand {

class Randomizer {
public:
    curandState *state;

    Randomizer(int X, int Y, int Z) {
        cudaError_t err = cudaMalloc( (void**)(&state), sizeof(curandState)*X*Y*Z );
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error allocating cuRAND state [%dx%dx%d]. ",X,Y,Z);
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }

        int3 ss_siz = make_int3(X,Y,Z);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,ss_siz.x,ss_siz.y,ss_siz.z);

        unsigned long long seed = clock();
        GpuKernelsRand::rand_setup<<<grd,blk>>>(state,seed,ss_siz);
    }

    void gen_normal(float*g_ptr,const float avg, const float std,const int3 ss_siz,cudaStream_t strm=0) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,ss_siz.x,ss_siz.y,ss_siz.z);
        GpuKernelsRand::gen_normal<<<grd,blk,0,strm>>>(g_ptr,state,avg,std,ss_siz);
    }

    void gen_normal(float*g_ptr,const float2*g_avg_std,const int3 ss_siz,cudaStream_t strm=0) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,ss_siz.x,ss_siz.y,ss_siz.z);
        GpuKernelsRand::gen_normal<<<grd,blk,0,strm>>>(g_ptr,state,g_avg_std,ss_siz);
    }

    ~Randomizer() {
        cudaFree(state);
    }
};

class BoxMullerRand {
    /// Based on:
    /// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
public:
    GPU::GArrUint32 current_state;

    BoxMullerRand(int X, int Y, int Z) {
        int32 numel = X*Y*Z;
        current_state.alloc(numel);

        uint32 *initial_state = new uint32[numel];
        srand((unsigned)clock());
        for(int i=0;i<numel;i++) initial_state[i] = rand();

        cudaError_t err = cudaMemcpy( (void*)(current_state.ptr), (const void*)initial_state, sizeof(uint32)*numel, cudaMemcpyHostToDevice);
        delete[] initial_state;
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error uploading random seed to CUDA memory. ");
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }

        GPU::sync();
    }

    void gen_normal(float*g_ptr,const float2*g_avg_std,const int3 ss_siz,cudaStream_t strm=0) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,ss_siz.x,ss_siz.y,ss_siz.z);
        uint32 seed = clock();
        GpuKernelsRand::gen_normal_box_muller<<<grd,blk,0,strm>>>(g_ptr,current_state.ptr,seed,g_avg_std,ss_siz);
    }
};

}

#endif /// GPU_RAND_H


