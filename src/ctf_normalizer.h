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

#ifndef CTF_NORMALIZER_H
#define CTF_NORMALIZER_H

#include "datatypes.h"
#include "particles.h"
#include "tomogram.h"
#include "stack_reader.h"
#include "gpu.h"
#include "gpu_fft.h"
#include "gpu_kernel.h"
#include "gpu_kernel_ctf.h"
#include "substack_crop.h"
#include "mrc.h"
#include "io.h"
#include "points_provider.h"
#include "svg.h"
#include "estimate_ctf_args.h"
#include <iostream>

class CtfNormalizer {
public:
    int3  ss_siz;
    int3  ss_buf;
    dim3  blk;
    dim3  grd;
    dim3  grd_buf;
    GPU::Stream       stream;
    GPU::GArrDouble   ss_acc;
    GPU::GArrDouble   ss_wgt;
    GPU::GArrSingle   ss_norm;
    GPU::GArrSingle3  ss_vec_r;
    GPU::GArrSingle   ss_ctf_ps;
    GPU::GArrSingle   ss_bufferA;
    GPU::GArrSingle   ss_bufferB;
    GPU::GArrSingle2  ss_fourier;
    GpuFFT::FFT2D     fft2;

    CtfNormalizer(int N,int K) {
        stream.configure();

        ss_siz  = make_int3((N/2)+1,N,K);
        ss_buf  = make_int3(N,N,K);
        blk     = GPU::get_block_size_2D();
        grd     = GPU::calc_grid_size(blk,ss_siz.x,ss_siz.y,ss_siz.z);
        grd_buf = GPU::calc_grid_size(blk,ss_buf.x,ss_buf.y,ss_buf.z);

        ss_acc.alloc(ss_siz.x*ss_siz.y*ss_siz.z);
        ss_wgt.alloc(ss_siz.x*ss_siz.y*ss_siz.z);
        ss_norm.alloc(ss_siz.x*ss_siz.y*ss_siz.z);
        ss_vec_r.alloc(ss_siz.x*ss_siz.y);
        ss_ctf_ps.alloc(ss_siz.x*ss_siz.y*ss_siz.z);
        ss_bufferA.alloc(ss_buf.x*ss_buf.y*ss_buf.z);
        ss_bufferB.alloc(ss_buf.x*ss_buf.y*ss_buf.z);
        ss_fourier.alloc(ss_siz.x*ss_siz.y*ss_siz.z);

        dim3 grd2D = GPU::calc_grid_size(blk,ss_siz.x,ss_siz.y,1);
        GpuKernelsCtf::create_vec_r<<<grd2D,blk,0,stream.strm>>>(ss_vec_r.ptr,ss_siz);

        fft2.alloc(ss_siz.x,ss_siz.y,ss_siz.z);
        fft2.set_stream(stream.strm);
    }

    void clear_acc() {
        ss_acc.clear(stream.strm);
        ss_wgt.clear(stream.strm);
        stream.sync();
    }

    void average(float*p_data,float2*p_factor) {
        remove_bg(p_data);
        exec_fft2(p_data);
        accumulate(p_factor);
        stream.sync();
    }

    void normal(float*p_data,float2*p_factor) {
        remove_bg(p_data);
        exec_fft2(p_data);
        normalize(p_factor);
        stream.sync();
    }

    void remove_bg(float*p_data) {
        float num = 0.0625f/2;
        float scl = 10.02548736875733f;
        GpuKernels::conv_gauss_1D_Y<<<grd_buf,blk,0,stream.strm>>>(ss_bufferA.ptr,p_data,num,scl,ss_buf);
        GpuKernels::conv_gauss_1D_X<<<grd_buf,blk,0,stream.strm>>>(ss_bufferB.ptr,ss_bufferA.ptr,num,scl,ss_buf,false);
        GpuKernels::substract    <<<grd_buf,blk,0,stream.strm>>>(p_data,ss_bufferB.ptr,ss_buf);
        GpuKernels::stk_hanning  <<<grd_buf,blk,0,stream.strm>>>(p_data,ss_buf);
    }

    void apply_wgt() {
        GpuKernelsCtf::divide<<<grd,blk,0,stream.strm>>>(ss_ctf_ps.ptr,ss_acc.ptr,ss_wgt.ptr,ss_siz);
    }

    void download(single*c_rslt) {
        GPU::download_async(c_rslt,ss_ctf_ps.ptr,ss_siz.x*ss_siz.y*ss_siz.z,stream.strm);
    }

protected:
    void exec_fft2(float*p_data) {
        fft2.exec(ss_fourier.ptr,p_data);
        GpuKernels::fftshift2D<<<grd,blk,0,stream.strm>>>(ss_fourier.ptr,ss_siz);
    }

    void normalize(float2*p_factor) {
        GpuKernelsCtf::ctf_normalize_ps<<<grd,blk,0,stream.strm>>>(ss_acc.ptr,ss_wgt.ptr,ss_fourier.ptr,p_factor,ss_siz);
    }

    void accumulate(float2*p_factor) {
        GpuKernelsCtf::accumulate_ps<<<grd,blk,0,stream.strm>>>(ss_acc.ptr,ss_wgt.ptr,ss_fourier.ptr,p_factor,ss_siz);
    }

    void accumulate() {
        GpuKernelsCtf::accumulate<<<grd,blk,0,stream.strm>>>(ss_acc.ptr,ss_wgt.ptr,ss_ctf_ps.ptr,ss_siz);
    }

};

#endif /// CTF_NORMALIZER_H

