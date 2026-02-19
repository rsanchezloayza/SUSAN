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

#ifndef REF_ALI_H
#define REF_ALI_H

#include "datatypes.h"
#include "math_cpu.h"
#include "gpu.h"
#include "gpu_fft.h"
#include "gpu_rand.h"
#include "gpu_kernel.h"
#include "gpu_kernel_ctf.h"
#include "gpu_kernel_vol.h"
//#include "substack_crop.h"
//#include "mrc.h"
//#include "io.h"
#include "points_provider.h"
//#include "angles_symmetry.h"
//#include "aligner_args.h"
#include <iostream>

class RadialAverager {
public:
    int N;
    int M;
    int maxK;

    GPU::GArrSingle  rad_avg;
    GPU::GArrSingle  rad_wgt;

    GPU::GArrSingle  std_acc;

    RadialAverager(int m,int n,int k)  {
        N  = n;
        M  = m;
        maxK = k;

        rad_avg.alloc(M*maxK);
        rad_wgt.alloc(M*maxK);

        std_acc.alloc(maxK);
    }

    void preset_FRC_vol(GPU::GArrSingle2&data) {
        rad_avg.clear();
        rad_wgt.clear();
        GPU::sync();
        int3 ss = make_int3(M,N,N);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,N);
        dim3 blk_frc = GPU::get_block_size_2D();
        dim3 grd_frc = GPU::calc_grid_size(blk,1,1,1);
        GpuKernels::radial_frc_avg_vol<<<grd,blk>>>(rad_avg.ptr,data.ptr,ss);
        GpuKernels::radial_frc_acc<<<grd_frc,blk_frc>>>(rad_avg.ptr,0,0,1,M);
        GpuKernels::radial_frc_norm_vol<<<grd,blk>>>(data.ptr,rad_avg.ptr,ss);
    }

    void preset_FRC(GPU::GArrSingle2&data,int k,GPU::Stream&stream) {
        rad_avg.clear(stream.strm);
        rad_wgt.clear(stream.strm);
        stream.sync();
        int3 ss = make_int3(M,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        dim3 blk_frc = GPU::get_block_size_2D();
        dim3 grd_frc = GPU::calc_grid_size(blk,k,1,1);
        GpuKernels::radial_frc_avg_stk<<<grd,blk,0,stream.strm>>>(rad_avg.ptr,data.ptr,ss);
        GpuKernels::radial_frc_acc<<<grd_frc,blk_frc,0,stream.strm>>>(rad_avg.ptr,0,0,k,M);
        GpuKernels::radial_frc_norm_stk<<<grd,blk,0,stream.strm>>>(data.ptr,rad_avg.ptr,ss);
    }

    void calculate_FRC(GPU::GArrSingle2&data,int k,GPU::Stream&stream) {
        rad_avg.clear(stream.strm);
        rad_wgt.clear(stream.strm);
        stream.sync();
        int3 ss = make_int3(M,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        GpuKernels::radial_frc_avg_stk<<<grd,blk,0,stream.strm>>>(rad_avg.ptr,data.ptr,ss);
    }

    void apply_FRC(GPU::GArrSingle2&data,int k,GPU::Stream&stream) {
        CtfConst ctf_const = {0};
        float2   ssnr      = {0};
        apply_FRC(data,ctf_const,ssnr,k,stream);
    }

    void apply_FRC(GPU::GArrSingle2&data,const CtfConst ctf_const,float2 ssnr,int k,GPU::Stream&stream) {
        single ssnr_f = -100*ssnr.x/(N*ctf_const.apix);
        single ssnr_s = pow(10,3*ssnr.y);
        int3 ss = make_int3(M,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        dim3 blk_frc = GPU::get_block_size_2D();
        dim3 grd_frc = GPU::calc_grid_size(blk,k,1,1);
        GpuKernels::radial_frc_acc<<<grd_frc,blk_frc,0,stream.strm>>>(rad_avg.ptr,ssnr_f,ssnr_s,k,M);
        GpuKernels::radial_frc_norm_stk<<<grd,blk,0,stream.strm>>>(data.ptr,rad_avg.ptr,ss);
    }

    void normalize_ctf(GPU::GArrSingle&ctf,int k,GPU::Stream&stream) {
        rad_avg.clear(stream.strm);
        rad_wgt.clear(stream.strm);
        stream.sync();
        int3 ss = make_int3(M,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        GpuKernels::get_energy_stk<<<grd,blk,0,stream.strm>>>(rad_avg.ptr,ctf.ptr,ss);
        GpuKernels::divide_energy_stk<<<grd,blk,0,stream.strm>>>(ctf.ptr,rad_avg.ptr,ss);
    }

    void normalize_stacks(GPU::GArrSingle2&data,float3 bandpass,int k,GPU::Stream&stream) {
        std_acc.clear(stream.strm);
        int3 ss  = make_int3(M,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        float scale = N;
        scale  = N / (2 * bandpass.y );
        scale *= scale;
        GpuKernels::get_std_from_fourier_stk<<<grd,blk,0,stream.strm>>>(std_acc.ptr,data.ptr,bandpass,ss);
        GpuKernels::apply_std_to_fourier_stk<<<grd,blk,0,stream.strm>>>(data.ptr,std_acc.ptr,ss);
        GpuKernels::divide<<<grd,blk,0,stream.strm>>>(data.ptr,scale,ss);
    }

};

class AliSubstack {
public:
    int N;
    int M;
    int P;
    int maxK;
    int NP;
    int MP;

    GPU::GArrSingle  ss_padded;
    GPU::GArrSingle3 ss_vec_r;
    GPU::GArrSingle2 ss_fourier;
    GPU::GArrSingle  rad_avg;
    GPU::GArrSingle  rad_wgt;

    GpuFFT::FFT2D          fft2;
    GpuRand::BoxMullerRand rand;

    AliSubstack(int m,int n,int k,int p,GPU::Stream&stream) : rand(n+p,n+p,k) {
        N  = n;
        M  = m;
        P  = p;
        NP = n+p;
        MP = (NP/2)+1;
        maxK = k;

        ss_padded.alloc(NP*NP*maxK);
        ss_fourier.alloc(MP*NP*maxK);

        rad_avg.alloc(MP*maxK);
        rad_wgt.alloc(MP*maxK);

        ss_vec_r.alloc(MP*NP);

        int3 ss_siz = make_int3(MP,NP,1);
        dim3 blk2D = GPU::get_block_size_2D();
        dim3 grd2D = GPU::calc_grid_size(blk2D,ss_siz.x,ss_siz.y,ss_siz.z);
        GpuKernelsCtf::create_vec_r<<<grd2D,blk2D,0,stream.strm>>>(ss_vec_r.ptr,ss_siz);

        fft2.alloc(MP,NP,maxK);
        fft2.set_stream(stream.strm);
    }

    void pad_zero(GPU::Stream&stream) {
        ss_padded.clear(stream.strm);
    }

    void pad_normal(GPU::GArrSingle2&g_avg_std,int k,GPU::Stream&stream) {
        int3 ss_pad = make_int3(NP,NP,k);
        rand.gen_normal(ss_padded.ptr,g_avg_std.ptr,ss_pad,stream.strm);
    }

    void add_data(GPU::GArrSingle&g_data,GPU::GArrProj2D&g_ali,int k,GPU::Stream&stream) {
        int3 pad = make_int3(P/2,P/2,0);
        int3 ss_raw = make_int3(N,N,k);
        int3 ss_pad = make_int3(NP,NP,k);
        int3 ss_fou = make_int3(MP,NP,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd_raw = GPU::calc_grid_size(blk,N,N,k);
        dim3 grd_pad = GPU::calc_grid_size(blk,NP,NP,k);
        dim3 grd_fou = GPU::calc_grid_size(blk,MP,NP,k);
        GpuKernels::load_pad<<<grd_raw,blk,0,stream.strm>>>(ss_padded.ptr,g_data.ptr,pad,ss_raw,ss_pad);
        GpuKernels::fftshift2D<<<grd_pad,blk,0,stream.strm>>>(ss_padded.ptr,ss_pad);
        fft2.exec(ss_fourier.ptr,ss_padded.ptr);
        GpuKernels::sampling_correction_2D<<<grd_fou,blk,0,stream.strm>>>(ss_fourier.ptr,2.0,ss_fou);
        GpuKernels::fftshift2D<<<grd_fou,blk,0,stream.strm>>>(ss_fourier.ptr,ss_fou);
        GpuKernels::subpixel_shift<<<grd_fou,blk,0,stream.strm>>>(ss_fourier.ptr,g_ali.ptr,ss_fou);
        GpuKernels::divide<<<grd_fou,blk,0,stream.strm>>>(ss_fourier.ptr,NP*NP,ss_fou);
    }

    void mask_small_ctf(GPU::GArrSingle&ctf_wgt,int k,GPU::Stream&stream) {
        int3 ss = make_int3(MP,NP,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
        GpuKernelsCtf::mask_small_ctf<<<grd,blk,0,stream.strm>>>(ss_fourier.ptr,ctf_wgt.ptr,ss);
    }

    void correct_phase_flip(const CtfConst ctf_const,GPU::GArrSingle&ctf_wgt,GPU::GArrDefocus&g_def,float3 bandpass,int k,GPU::Stream&stream) {
        int3 ss = make_int3(MP,NP,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
        GpuKernelsCtf::correct_stk_phase_flip<<<grd,blk,0,stream.strm>>>(ss_fourier.ptr,ctf_wgt.ptr,g_def.ptr,bandpass,ctf_const,ss);
    }

    void correct_wiener(const CtfConst ctf_const,GPU::GArrSingle&ctf_wgt,GPU::GArrDefocus&g_def,float3 bandpass,int k,GPU::Stream&stream) {
        int3 ss = make_int3(MP,NP,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
        GpuKernelsCtf::correct_stk_wiener<<<grd,blk,0,stream.strm>>>(ss_fourier.ptr,ctf_wgt.ptr,g_def.ptr,bandpass,ctf_const,ss);
    }

    void correct_wiener_ssnr(const CtfConst ctf_const,GPU::GArrSingle&ctf_wgt,GPU::GArrDefocus&g_def,float3 bandpass,float2 ssnr,int k,GPU::Stream&stream) {
        // float2  ssnr; /// x=F; y=S;
        single ssnr_f = -100*ssnr.x/(NP*ctf_const.apix);
        single ssnr_s = pow(10,3*ssnr.y);
        int3 ss = make_int3(MP,NP,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
        GpuKernelsCtf::correct_stk_wiener_ssnr<<<grd,blk,0,stream.strm>>>(ss_fourier.ptr,ctf_wgt.ptr,g_def.ptr,ssnr_f,ssnr_s,bandpass,ctf_const,ss);
    }

    void whitening_filter(int k,GPU::Stream&stream) {
        whitening_filter(ss_fourier,k,stream);
    }

    void whitening_filter(GPU::GArrSingle2&ss_data,int k,GPU::Stream&stream) {
        rad_avg.clear(stream.strm);
        rad_wgt.clear(stream.strm);
        stream.sync();
        int3 ss = make_int3(MP,NP,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
        GpuKernels::radial_ps_avg<<<grd,blk,0,stream.strm>>>(rad_avg.ptr,rad_wgt.ptr,ss_data.ptr,ss);
        stream.sync();
        GpuKernels::radial_ps_norm<<<grd,blk,0,stream.strm>>>(ss_data.ptr,rad_avg.ptr,rad_wgt.ptr,ss);
        stream.sync();
    }

    void preset_FSC(GPU::GArrSingle2&ss_data,int k,GPU::Stream&stream) {
        rad_avg.clear(stream.strm);
        rad_wgt.clear(stream.strm);
        stream.sync();
        int3 ss = make_int3(MP,NP,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
        dim3 blk_frc = GPU::get_block_size_2D();
        dim3 grd_frc = GPU::calc_grid_size(blk,k,1,1);
        GpuKernels::radial_frc_avg_stk<<<grd,blk,0,stream.strm>>>(rad_avg.ptr,ss_data.ptr,ss);
        GpuKernels::radial_frc_acc<<<grd_frc,blk_frc,0,stream.strm>>>(rad_avg.ptr,0,0,k,MP);
        GpuKernels::radial_frc_norm_stk<<<grd,blk,0,stream.strm>>>(ss_data.ptr,rad_avg.ptr,ss);
    }

    void preset_FSC(int k,GPU::Stream&stream) {
        preset_FSC(ss_fourier,k,stream);
    }

    void norm_complex(GPU::GArrSingle2&ss_data,int k,GPU::Stream&stream) {
        int3 ss = make_int3(MP,NP,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
        GpuKernels::norm_complex<<<grd,blk,0,stream.strm>>>(ss_data.ptr,ss);
        GpuKernels::apply_radial_wgt<<<grd,blk,0,stream.strm>>>(ss_data.ptr,k,MP,ss);
    }

    void apply_radial_wgt(float w_total,float crowther_limit,int k,GPU::Stream&stream) {
        int3 ss = make_int3(MP,NP,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
        float limit = fminf(crowther_limit,MP);
        GpuKernels::apply_radial_wgt<<<grd,blk,0,stream.strm>>>(ss_fourier.ptr,w_total,limit,ss);
    }

};

class AliData {
public:
    Vec3           *c_pts;
    GPU::GArrProj2D g_ali;
    GPU::GArrVec3   g_pts;
    GPU::GArrSingle g_cc;
    float          *c_cc;
    uint32 M;
    uint32 N;
    uint32 max_K;
    uint32 n_pts;

    GPU::GArrSingle2 prj_c;
    GPU::GArrSingle  prj_r;

    GpuFFT::IFFT2D ifft2;

    GPU::GTex2DSingle prj_tex;

    AliData(uint32 m, uint32 n, uint32 n_K,const float4&off_params,int off_type,GPU::Stream&stream) {
        M = m;
        N = n;
        max_K = n_K;
        g_ali.alloc(max_K);
        prj_c.alloc(M*N*max_K);
        prj_r.alloc(N*N*max_K);

        prj_tex.alloc(n,n,n_K);
        
        ifft2.alloc(M,N,max_K);
        ifft2.set_stream(stream.strm);

        if( off_type == ELLIPSOID )
            c_pts = PointsProvider::ellipsoid(n_pts,off_params.x,off_params.y,off_params.z,off_params.w);
        if( off_type == CYLINDER )
            c_pts = PointsProvider::cylinder(n_pts,off_params.x,off_params.y,off_params.z,off_params.w);
        if( off_type == CUBOID )
            c_pts = PointsProvider::cuboid(n_pts,off_params.x,off_params.y,off_params.z,off_params.w);
        if( off_type == CIRCLE )
            c_pts = PointsProvider::circle(n_pts,off_params.x,off_params.y,off_params.w);
        if( off_type == RECTANGLE )
            c_pts = PointsProvider::rectangle(n_pts,off_params.x,off_params.y,off_params.w);
        
        if( (off_type == CIRCLE)  || (off_type == RECTANGLE) ) {
            c_cc = new float[n_pts*max_K];
            g_cc.alloc(n_pts*max_K);
        }
        else {
            c_cc = new float[n_pts];
            g_cc.alloc(n_pts);
        }
        g_pts.alloc(n_pts);
        cudaMemcpy((void*)g_pts.ptr,(const void*)c_pts,sizeof(Vec3)*n_pts,cudaMemcpyHostToDevice);
    }

    ~AliData() {
        delete [] c_cc;
        delete [] c_pts;
    }

    void pre_rotate_reference(Rot33&R,GPU::GArrProj2D&ali_in,int k,GPU::Stream&stream) {
        dim3 blk;
        dim3 grd;
        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(9*k,1024);
        grd.y = 1;
        grd.z = 1;
        cudaMemcpyAsync((void*)g_ali.ptr,(const void*)ali_in.ptr,sizeof(Proj2D)*k,cudaMemcpyDeviceToDevice,stream.strm);
        stream.sync();
        GpuKernels::rotate_post<<<grd,blk,0,stream.strm>>>(ali_in.ptr,R,g_ali.ptr,k);
    }

    void rotate_reference(Rot33&R,GPU::GArrProj2D&ali_in,int k,GPU::Stream&stream) {
        dim3 blk;
        dim3 grd;
        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(9*k,1024);
        grd.y = 1;
        grd.z = 1;

        GpuKernels::rotate_post<<<grd,blk,0,stream.strm>>>(g_ali.ptr,R,ali_in.ptr,k);
    }

    void rotate_projections(Rot33&R,GPU::GArrProj2D&ali_in,int k,GPU::Stream&stream) {
        dim3 blk;
        dim3 grd;
        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(9*k,1024);
        grd.y = 1;
        grd.z = 1;

        GpuKernels::rotate_pre<<<grd,blk,0,stream.strm>>>(g_ali.ptr,R,ali_in.ptr,k);
    }

    void project(GPU::GTex3DSingle2&ref,float3 bandpass,int k,GPU::Stream&stream) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        GpuKernelsVol::extract_stk<<<grd,blk,0,stream.strm>>>(prj_c.ptr,ref.texture,g_ali.ptr,bandpass,M,N,k);
    }

    void invert_fourier(int k,GPU::Stream&stream) {
        int3 ss_fou = make_int3(M,N,k);
        int3 ss_pad = make_int3(N,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd_f = GPU::calc_grid_size(blk,M,N,k);
        dim3 grd_r = GPU::calc_grid_size(blk,N,N,k);
        GpuKernels::fftshift2D<<<grd_f,blk,0,stream.strm>>>(prj_c.ptr,ss_fou);
        GpuKernels::sampling_correction_2D<<<grd_f,blk,0,stream.strm>>>(prj_c.ptr,0.5,ss_fou);
        ifft2.exec(prj_r.ptr,prj_c.ptr);
        GpuKernels::fftshift2D<<<grd_r,blk,0,stream.strm>>>(prj_r.ptr,ss_pad);
    }

    void scale(float scale,int k,GPU::Stream&stream) {
        int3 ss = make_int3(N,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,N,N,k);
        GpuKernels::divide<<<grd,blk,0,stream.strm>>>(prj_r.ptr,scale,ss);
    }

    void apply_bandpass(float3 bandpass,int k,GPU::Stream&stream) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        GpuKernels::apply_bandpass_fourier<<<grd,blk,0,stream.strm>>>(prj_c.ptr,bandpass,M,N,k);
    }

    void apply_bandpass(GPU::GArrDefocus&p_def,float3 bandpass,int k,GPU::Stream&stream) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        GpuKernels::apply_bandpass_fourier<<<grd,blk,0,stream.strm>>>(prj_c.ptr,p_def.ptr,bandpass,M,N,k);
    }

    void apply_bandpass(const CtfConst ctf_const,GPU::GArrDefocus&p_def,float3 bandpass,int k,GPU::Stream&stream) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        GpuKernelsCtf::apply_bandpass_fourier<<<grd,blk,0,stream.strm>>>(prj_c.ptr,ctf_const,p_def.ptr,bandpass,M,N,k);
    }

    void multiply(GPU::GArrSingle&p_wgt,int k,GPU::Stream&stream) {
        int3 ss = make_int3(M,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        GpuKernels::multiply<<<grd,blk,0,stream.strm>>>(prj_c.ptr,p_wgt.ptr,ss);
    }

    void multiply(GPU::GArrSingle2&p_data,int k,GPU::Stream&stream) {
        int3 ss = make_int3(M,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,k);
        GpuKernels::multiply<<<grd,blk,0,stream.strm>>>(prj_c.ptr,p_data.ptr,ss);
    }

    void sparse_reconstruct(GPU::GArrProj2D&ali,int dilate,int k,GPU::Stream&stream) {
        int3 ss = make_int3(N,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,N,N,k);
        if( dilate == 1 )
            GpuKernels::load_surf_dilate_1<<<grd,blk,0,stream.strm>>>(prj_tex.surface,prj_r.ptr,ss);
        else if( dilate == 2 )
            GpuKernels::load_surf_dilate_2<<<grd,blk,0,stream.strm>>>(prj_tex.surface,prj_r.ptr,ss);
        else if( dilate == 3 )
            GpuKernels::load_surf_dilate_3<<<grd,blk,0,stream.strm>>>(prj_tex.surface,prj_r.ptr,ss);
        else
            GpuKernels::load_surf<<<grd,blk,0,stream.strm>>>(prj_tex.surface,prj_r.ptr,ss);

        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(n_pts,1024);
        grd.y = 1;
        grd.z = 1;
        GpuKernelsVol::reconstruct_pts<<<grd,blk,0,stream.strm>>>(g_cc.ptr,ali.ptr,prj_tex.texture,g_pts.ptr,n_pts,N,k);
        GPU::download_async(c_cc,g_cc.ptr,n_pts,stream.strm);
    }

    void extract_cc(float*p_cc,int*p_ix,GPU::GArrProj2D&ali,int k,GPU::Stream&stream) {
        dim3 blk;
        dim3 grd;
        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(n_pts,1024);
        grd.y = 1;
        grd.z = k;
        GpuKernelsVol::extract_pts<<<grd,blk,0,stream.strm>>>(g_cc.ptr,prj_r.ptr,ali.ptr,g_pts.ptr,n_pts,N,k);
        GPU::download_async(c_cc,g_cc.ptr,n_pts*k,stream.strm);
        stream.sync();
        for(int i=0;i<k;i++) {
            get_max_cc(p_cc[i],p_ix[i],c_cc+i*n_pts);
        }
    }

    void get_max_cc(float&max_cc,int&max_idx,const float*p_data) {
        max_idx = 0;
        max_cc = p_data[max_idx];
        for(int i=0;i<n_pts;i++) {
            if( max_cc < p_data[i] ) {
                max_cc = p_data[i];
                max_idx = i;
            }
        }
    }

    float get_sum_cc(const float*p_data) {
        return Math::sum_vec(p_data,n_pts);
    }

    void aggregate_avg_std(float&avg,float&std,float&count,const float*p_data) {
        count += n_pts;
        for(int i=0;i<n_pts;i++) {
            float tmp = p_data[i];
            avg += tmp;
            std += (tmp*tmp);
        }
    }
};

class AliRef {
public:
    GPU::GTex3DSingle2 ref;
    int N;
    int M;

    void allocate(GPU::GArrSingle2&g_fou,int m,int n) {
        M = m;
        N = n;
        ref.alloc(M,N,N);

        bool should_conjugate = true;

        int3 siz = make_int3(M,N,N);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,N);
        GpuKernels::load_surf_3<<<grd,blk>>>(ref.surface,g_fou.ptr,siz,should_conjugate);
    }
};

class ProjectionData {
public:
    GPU::GArrProj2D g_ali;
    uint32 M;
    uint32 N;
    uint32 P;
    uint32 MP;
    uint32 NP;
    uint32 max_K;
    uint32 n_pts;

    GPU::GArrSingle2 prj_c;
    GPU::GArrSingle  prj_r;

    GpuFFT::IFFT2D ifft2;

    GPU::GTex2DSingle prj_tex;

    ProjectionData(uint32 m, uint32 n, uint32 p, uint32 n_K,GPU::Stream&stream) {
        M  = m;
        N  = n;
        P  = p;
        NP = n*p;
        MP = m*p;
        max_K = n_K;
        g_ali.alloc(max_K);
        prj_c.alloc(MP*NP*max_K);
        prj_r.alloc(NP*NP*max_K);

        prj_tex.alloc(NP,NP,n_K);

        ifft2.alloc(MP,NP,max_K);
        ifft2.set_stream(stream.strm);
    }

    ~ProjectionData() {
    }

    void pre_rotate_reference(Rot33&R,GPU::GArrProj2D&ali_in,int k,GPU::Stream&stream) {
        dim3 blk;
        dim3 grd;
        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(9*k,1024);
        grd.y = 1;
        grd.z = 1;
        cudaMemcpyAsync((void*)g_ali.ptr,(const void*)ali_in.ptr,sizeof(Proj2D)*k,cudaMemcpyDeviceToDevice,stream.strm);
        stream.sync();
        GpuKernels::rotate_post<<<grd,blk,0,stream.strm>>>(ali_in.ptr,R,g_ali.ptr,k);
    }

    void rotate_reference(Rot33&R,GPU::GArrProj2D&ali_in,int k,GPU::Stream&stream) {
        dim3 blk;
        dim3 grd;
        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(9*k,1024);
        grd.y = 1;
        grd.z = 1;

        GpuKernels::rotate_post<<<grd,blk,0,stream.strm>>>(g_ali.ptr,R,ali_in.ptr,k);
    }

    void rotate_projections(Rot33&R,GPU::GArrProj2D&ali_in,int k,GPU::Stream&stream) {
        dim3 blk;
        dim3 grd;
        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(9*k,1024);
        grd.y = 1;
        grd.z = 1;

        GpuKernels::rotate_pre<<<grd,blk,0,stream.strm>>>(g_ali.ptr,R,ali_in.ptr,k);
    }

    void project(GPU::GTex3DSingle2&ref,float3 bandpass,int k,GPU::Stream&stream) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
        GpuKernelsVol::extract_stk<<<grd,blk,0,stream.strm>>>(prj_c.ptr,ref.texture,g_ali.ptr,bandpass,MP,NP,k);
    }

    void invert_fourier(int k,GPU::Stream&stream) {
        int3 ss_fou = make_int3(MP,NP,k);
        int3 ss_pad = make_int3(NP,NP,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd_f = GPU::calc_grid_size(blk,MP,NP,k);
        dim3 grd_r = GPU::calc_grid_size(blk,NP,NP,k);
        GpuKernels::fftshift2D<<<grd_f,blk,0,stream.strm>>>(prj_c.ptr,ss_fou);
        ifft2.exec(prj_r.ptr,prj_c.ptr);
        GpuKernels::fftshift2D<<<grd_r,blk,0,stream.strm>>>(prj_r.ptr,ss_pad);
    }

    void extract_real(GPU::GArrSingle&p_out,int k,GPU::Stream&stream) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,N,N,k);
        int3 ss_pad = make_int3(NP,NP,k);
        int3 ss_out = make_int3(N ,N ,k);
        GpuKernels::remove_pad_stk<<<grd,blk,0,stream.strm>>>(p_out.ptr,prj_r.ptr,P,ss_out,ss_pad);
    }

    void multiply(GPU::GArrSingle&p_wgt,int k,GPU::Stream&stream) {
        int3 ss = make_int3(MP,NP,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
        GpuKernels::multiply<<<grd,blk,0,stream.strm>>>(prj_c.ptr,p_wgt.ptr,ss);
    }

    void scale(float scale,int k,GPU::Stream&stream) {
        int3 ss = make_int3(N,N,k);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,NP,NP,k);
        GpuKernels::divide<<<grd,blk,0,stream.strm>>>(prj_r.ptr,scale,ss);
    }

    void apply_bandpass(float3 bandpass,int k,GPU::Stream&stream) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,MP,NP,k);
        GpuKernels::apply_bandpass_fourier<<<grd,blk,0,stream.strm>>>(prj_c.ptr,bandpass,MP,NP,k);
    }
};



#endif /// REF_ALI_H


