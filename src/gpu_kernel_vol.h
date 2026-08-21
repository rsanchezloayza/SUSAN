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

#ifndef GPU_KERNEL_VOL_H
#define GPU_KERNEL_VOL_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "datatypes.h"

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cufft.h"

#include "gpu.h"
#include "gpu_kernel.h"

using namespace GpuKernels;

namespace GpuKernelsVol {

__device__ void  rot_pt(float&x,float&y,float&z,const Rot33&R,const Vec3&in) {
    x = R.xx*in.x + R.xy*in.y + R.xz*in.z;
    y = R.yx*in.x + R.yy*in.y + R.yz*in.z;
    z = R.zx*in.x + R.zy*in.y + R.zz*in.z;
}

__device__ void  rot_pt_XY(Vec3&out,const Rot33&R,const Vec3&in) {
    out.x = R.xx*in.x + R.xy*in.y;
    out.y = R.yx*in.x + R.yy*in.y;
    out.z = R.zx*in.x + R.zy*in.y;
}

__device__ void  rot_inv_pt(float&x,float&y,float&z,const Rot33&R,const Vec3&in) {
    x = R.xx*in.x + R.yx*in.y + R.zx*in.z;
    y = R.xy*in.x + R.yy*in.y + R.zy*in.z;
    z = R.xz*in.x + R.yz*in.y + R.zz*in.z;
}

__device__ float rot_inv_pt_Z(const Rot33&R,const Vec3&in) {
    return R.xz*in.x + R.yz*in.y + R.zz*in.z;
}

__device__ void  rot_inv_pt_XY(float&x,float&y,const Rot33&R,const Vec3&in) {
    x = R.xx*in.x + R.yx*in.y + R.zx*in.z;
    y = R.xy*in.x + R.yy*in.y + R.zy*in.z;
}

__device__ bool  get_mirror_index(int&x,int&y,int&z,const int M,const int N) {
    if( x < 0 ) {
        x = -x;
        y = N - y; if( y >= N ) y = 0;
        z = N - z; if( z >= N ) z = 0;
    }

    if( y>=0 && z >= 0 && x<M && y<N && z<N )
        return true;
    else
        return false;
}

__device__ bool  get_mirror_index(int&x,int&y,int&z,bool&was_inverted,const int M,const int N) {
    was_inverted = (x<0) ? true : false;
    return get_mirror_index(x,y,z,M,N);
}

__device__ int get_ix_3d(bool&should_read,bool&should_conj,const int x,const int y,const int z,const int M,const int N) {
    int wx=x;
    int wy=y;
    int wz=z;
    should_conj = false;
        should_read = false;

    if(x<0) {
        wx = -wx;
        wy = -wy;
        wz = -wz;
        should_conj = true;
    }

    wy += N/2;
    wz += N/2;

        if( wx < M && wy < N && wz < N )
            should_read = true;

    return wx + wy*M + wz*M*N;
}

////////////////////////////////////////////////////////////////////////

__global__ void insert_stk(double2*p_acc,double*p_wgt,
                           cudaTextureObject_t ss_stk, cudaTextureObject_t ss_wgt, const Proj2D*pTlt,
                           const float3 bandpass,const int M, const int N, const int K)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < N ) {

        Vec3 pt;
        pt.x = ss_idx.x;
        pt.y = ss_idx.y - N/2;
        pt.z = ss_idx.z - N/2;

        float R = sqrt( pt.x*pt.x + pt.y*pt.y + pt.z*pt.z );
        float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

        if( bp > 0.025 ) {

            bool should_add = false;
            double  wgt = 0;
            double2 val = {0,0};
            float x,y,z;

            for(int k=0; k<K; k++ ) {
                if( pTlt[k].w != 0 ){
                    z = rot_inv_pt_Z(pTlt[k].R,pt);
                    if( z >= 0 && z<= 1 ) {
                        should_add = true;
                        bool should_conj = false;
                        rot_inv_pt_XY(x,y,pTlt[k].R,pt);
                        if(x<0){
                            x = -x;
                            y = -y;
                            should_conj = true;
                        }
                        float2 read_stk = tex2DLayered<float2>(ss_stk, x+0.5, y+N/2+0.5, k);
                        if(should_conj) {
                            read_stk.y = -read_stk.y;
                        }
                        val.x += pTlt[k].w*(1-z)*read_stk.x;
                        val.y += pTlt[k].w*(1-z)*read_stk.y;
                        float  read_wgt = tex2DLayered<float >(ss_wgt, x+0.5, y+N/2+0.5, k);
                        wgt   += fabsf(pTlt[k].w)*(1-z)*read_wgt;

                    }
                }
            }

            if( should_add ) {
                long idx = ss_idx.x + ss_idx.y*M + ss_idx.z*M*N;
                double2 tmp = p_acc[ idx ];
                tmp.x += val.x;
                tmp.y += val.y;
                p_acc[ idx ]  = tmp;
                p_wgt[ idx ] += wgt;
            }

        }
    }

}

__global__ void insert_stk_atomic(double2*p_acc,double*p_wgt,
                                  cudaTextureObject_t ss_stk, cudaTextureObject_t ss_wgt, const Proj2D*pTlt,
                                  const float3 bandpass,const int M, const int N, const int K)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x >= M || ss_idx.y >= N || ss_idx.z >= K )
        return;

    if( pTlt[ss_idx.z].w == 0 )
        return;

    float Nh = float(N)/2;
    Vec3 pt;
    pt.x = ss_idx.x;
    pt.y = ss_idx.y - Nh;
    pt.z = 0;

    float R = sqrt( pt.x*pt.x + pt.y*pt.y );
    float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

    if( bp <= 1e-3f )
        return;

    float2 val     = tex2DLayered<float2>(ss_stk, float(ss_idx.x)+0.5, float(ss_idx.y)+0.5, ss_idx.z);
    float  ctf_wgt = tex2DLayered<float >(ss_wgt, float(ss_idx.x)+0.5, float(ss_idx.y)+0.5, ss_idx.z);
    float  prj_w   = pTlt[ss_idx.z].w * bp;

    float x,y,z;
    rot_pt(x,y,z,pTlt[ss_idx.z].R,pt);

    int ix0 = floorf(x);
    int iy0 = floorf(y);
    int iz0 = floorf(z);

    #pragma unroll
    for(int dz=0; dz<=1; dz++) {
        int iz = iz0 + dz;
        float wz = 1.0f - fabsf(z - iz);
        if( wz <= 0.0f ) continue;
        if( iz <  -Nh || iz >= Nh) continue;

        #pragma unroll
        for(int dy=0; dy<=1; dy++) {
            int iy = iy0 + dy;
            float wy = 1.0f - fabsf(y - iy);
            if( wy <= 0.0f ) continue;
            if( iy <  -Nh || iy >= Nh) continue;

            #pragma unroll
            for(int dx=0; dx<=1; dx++) {
                int ix = ix0 + dx;
                float wx = 1.0f - fabsf(x - ix);
                if( wx <= 0.0f ) continue;

                float lin_wgt = wx*wy*wz * prj_w;

                /// Insert into hermitian volume
                bool should_conj = false;
                int ix_h = ix;
                int iy_h = iy;
                int iz_h = iz;

                if (ix_h < 0) {
                    ix_h = -ix_h;
                    iy_h = -iy_h;
                    iz_h = -iz_h;
                    should_conj = true;
                }

                iy_h += Nh;
                iz_h += Nh;

                if (ix_h >= M) continue;
                if (iy_h < 0 || iy_h >= N) continue;
                if (iz_h < 0 || iz_h >= N) continue;

                long   idx = get_3d_idx(ix_h,iy_h,iz_h,M,N);
                float2 out = val;
                if( should_conj ) out.y = -out.y;

                atomic_Add( &(p_acc[idx].x) , lin_wgt*out.x   );
                atomic_Add( &(p_acc[idx].y) , lin_wgt*out.y   );
                atomic_Add( &(p_wgt[idx]  ) , fabsf(lin_wgt)*ctf_wgt );
            }
        }
    }
}

/// EXPERIMENTAL (SUSAN_SPLAT): angular-spread splatting.
///
/// Instead of inserting each Fourier pixel with a fixed-support kernel and hard-cutting at the
/// resolution limit, spread it over a gaussian whose width grows with the radius, and admit
/// every frequency. The model is the MAP insertion in RELION, where a slice is added at every
/// orientation weighted by its posterior probability.
///
/// A small orientation error w displaces a Fourier point p by d = w x p. For isotropic w with
/// per-axis std s_th, Cov(d) = s_th^2 * ( |p|^2 I - p p' ) = s_th^2 R^2 ( I - r r' ): rank 2,
/// zero radially, std s_th*R in both tangential directions. So the blur is purely tangential
/// and linear in R, and the radial width stays at the interpolation width always.
///
/// s_th is not measured; R_ref (the lowpass, tightened by the per-projection max_res) pins it.
/// Claiming a projection is good to R_ref means the angular error has not displaced anything
/// by more than one interpolation width at that shell, so s_th*R_ref = SIGMA_0, giving
/// s_th = SIGMA_0/R_ref. Below R_ref the induced blur is smaller than the interpolation kernel
/// itself and is already absorbed by it, which is why the width is flat there rather than
/// merely approximated as flat. The whole profile is then one clamp:
///
///     sigma_t(R) = clamp( splat_gain * SIGMA_0 * R / R_ref , SIGMA_0 , SIGMA_MAX )
///
/// splat_gain moves the crossover (R_ref/splat_gain) rather than the slope; 1 is the physical
/// anchor and 0 clamps to SIGMA_0 everywhere, i.e. plain trilinear. SIGMA_MAX is set by the
/// 5x5x5 support: at 1.0 the kernel keeps 97% of its mass inside the box, at 1.5 only 75%,
/// and the truncated remainder rings.
#define SPLAT_SIGMA_0   0.4f  /// closest match to trilinear over all sub-voxel offsets
#define SPLAT_SIGMA_MAX 1.0f
#define SPLAT_RAD       2     /// 5x5x5

/// Taps below this share of the (normalized) kernel are dropped. Scattered write traffic is
/// what this kernel costs, so the tap count is by far the most effective knob: at 5e-3 a source
/// pixel writes ~31 of its 125 taps and deposits 94% of the kernel mass. Dropping mass is
/// cheap because it is a uniform scale on p_acc and p_wgt, which cancels in their ratio; what
/// does not cancel is that the retained fraction varies with sigma_t (98.5% at 0.4 against 94%
/// at 1.0), so two projections with different max_res are weighted slightly differently where
/// they overlap. That spread is 4.5 points here, against 2.5 at 3e-3 and 9.3 at 1e-2.
///
/// Measured, per particle: 1e-6 -> 3e-3 was 335.6 -> 89.4 ms, and 3e-3 -> 5e-3 a further
/// 89.4 -> 73.2 ms. Verify a reconstruction before raising it again.
#define SPLAT_W_MIN     5e-3f

/// Warps per block in the main splat kernel, i.e. source pixels handled per block. Only affects
/// occupancy and scheduling granularity, not results.
#define SPLAT_WARPS     8


__device__ __forceinline__ float splat_sigma_t(const float R,const float R_ref,const float gain) {
    if( R_ref < 1.0f )
        return SPLAT_SIGMA_0;
    float s = gain*SPLAT_SIGMA_0*R/R_ref;
    return fminf( fmaxf(s,SPLAT_SIGMA_0), SPLAT_SIGMA_MAX );
}

/// Splatting runs as two kernels.
///
/// The shape below follows from where the time actually goes, which was measured rather than
/// assumed: of the original single-thread-per-pixel version, ~71% was scattered write traffic,
/// ~27% arithmetic, and only ~2.3% the atomics themselves. Restructuring for the traffic took
/// it from 73.2 to 21.4 ms per particle; against ordinary trilinear insertion at 4.9 ms, that
/// is ~4.3x rather than the ~50x the naive tap count suggests.
///
/// The per-source-pixel setup (landing point, sigma_t, and the 125-tap normalizer) is identical
/// for every tap, so it is hoisted into a pre-pass and read back by the main kernel. That keeps
/// the main kernel small enough to give a warp one source pixel instead of one thread, which is
/// what the memory pattern needs: the cost of this insertion is neither the atomics (2.3% of
/// runtime, measured) nor the arithmetic, it is that a store instruction fans out to 32 separate
/// transactions. Consecutive threads walk consecutive source pixels, whose landing points step
/// by only R.xx along the volume's fastest axis, so nothing coalesces.
///
/// One warp per source pixel fixes that. The 125 taps are a compact 5x5x5 neighbourhood, and
/// indexing them t = dx + 5*dy + 25*dz makes consecutive lanes carry consecutive dx, hence five
/// contiguous voxels at a time instead of 32 unrelated ones.
__global__ void splat_prepass(float4*p_geom,float*p_inorm,
                              const Proj2D*pTlt, const Defocus*pDef, const float splat_gain,
                              const float3 bandpass,const int M, const int N, const int K)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x >= M || ss_idx.y >= N || ss_idx.z >= K )
        return;

    long idx = get_3d_idx(ss_idx.x,ss_idx.y,ss_idx.z,M,N);

    /// inorm == 0 is how the main kernel learns to skip this source pixel.
    if( pTlt[ss_idx.z].w == 0 ) {
        p_inorm[idx] = 0;
        return;
    }

    int Nh = N/2;
    Vec3 pt;
    pt.x = ss_idx.x;
    pt.y = ss_idx.y - Nh;
    pt.z = 0;

    float R = sqrtf( pt.x*pt.x + pt.y*pt.y );

    /// No bandpass weight: the resolution limit acts through the kernel width instead. The
    /// corners of the box lie outside the nyquist sphere and would be discarded anyway.
    if( R > (float)Nh ) {
        p_inorm[idx] = 0;
        return;
    }

    /// Reference shell: the lowpass, tightened by the per-projection max_res when set. The CTF
    /// stage no longer cuts at either of them (see RecSubstack::set_*, use_max_res), so both
    /// survive here purely as the scale that sets the kernel width.
    float R_ref = bandpass.y;
    if( pDef[ss_idx.z].max_res > 0 )
        R_ref = fminf(R_ref,pDef[ss_idx.z].max_res);

    float sigma_t = splat_sigma_t(R,R_ref,splat_gain);

    float x,y,z;
    rot_pt(x,y,z,pTlt[ss_idx.z].R,pt);

    /// Centring on the nearest voxel rather than the floor keeps the captured mass at 97-99%
    /// whatever the sub-voxel offset; with floorf the support reaches 3.0 on one side and the
    /// truncated fraction swings with the landing position.
    float ix0 = rintf(x);
    float iy0 = rintf(y);
    float iz0 = rintf(z);

    float inv2_s0 = 1.0f/(2.0f*SPLAT_SIGMA_0*SPLAT_SIGMA_0);
    float inv2_st = 1.0f/(2.0f*sigma_t*sigma_t);

    /// The radial direction is undefined at the origin, where sigma_t is at the floor anyway.
    bool  aniso = (R > 1.0f);
    float rx = 0, ry = 0, rz = 0;
    if( aniso ) {
        rx = x/R;
        ry = y/R;
        rz = z/R;
    }

    /// The truncated, off-lattice gaussian does not sum to a constant, and the weight volume
    /// feeds a non-linear inversion downstream, so normalize the taps explicitly. Normalizing
    /// over the full support (not only the in-bounds taps) keeps the inserted mass consistent.
    float norm = 0;
    for(int dz=-SPLAT_RAD; dz<=SPLAT_RAD; dz++) {
        float ez = iz0 + float(dz) - z;
        for(int dy=-SPLAT_RAD; dy<=SPLAT_RAD; dy++) {
            float ey = iy0 + float(dy) - y;
            for(int dx=-SPLAT_RAD; dx<=SPLAT_RAD; dx++) {
                float ex = ix0 + float(dx) - x;
                float d2 = ex*ex + ey*ey + ez*ez;
                float e;
                if( aniso ) {
                    float d_par = ex*rx + ey*ry + ez*rz;
                    e = d_par*d_par*inv2_s0 + (d2 - d_par*d_par)*inv2_st;
                }
                else
                    e = d2*inv2_s0;
                norm += __expf(-e);
            }
        }
    }

    if( norm <= 1e-12f ) {
        p_inorm[idx] = 0;
        return;
    }

    p_geom[idx]  = make_float4(x,y,z,sigma_t);
    p_inorm[idx] = 1.0f/norm;
}

/// One warp per source pixel. blockDim must be (32, warps_per_block, 1).
__global__ void insert_stk_splat_atomic(double2*p_acc,double*p_wgt,
                                        cudaTextureObject_t ss_stk, cudaTextureObject_t ss_wgt,
                                        const float4*p_geom, const float*p_inorm,
                                        const Proj2D*pTlt,
                                        const int M, const int N, const int K)
{
    int lane = threadIdx.x;
    int i    = blockIdx.x*blockDim.y + threadIdx.y;
    int j    = blockIdx.y;
    int k    = blockIdx.z;

    if( i >= M || j >= N || k >= K )
        return;

    long  sidx  = get_3d_idx(i,j,k,M,N);
    float inorm = p_inorm[sidx];
    if( inorm <= 0 )
        return;

    float4 geom = p_geom[sidx];
    float  x = geom.x, y = geom.y, z = geom.z;
    float  sigma_t = geom.w;

    int   Nh = N/2;
    float pty = float(j) - float(Nh);
    float R   = sqrtf( float(i)*float(i) + pty*pty );

    float2 val     = tex2DLayered<float2>(ss_stk, float(i)+0.5, float(j)+0.5, k);
    float  ctf_wgt = tex2DLayered<float >(ss_wgt, float(i)+0.5, float(j)+0.5, k);
    float  prj_w   = pTlt[k].w;

    int ix0 = (int)rintf(x);
    int iy0 = (int)rintf(y);
    int iz0 = (int)rintf(z);

    float inv2_s0 = 1.0f/(2.0f*SPLAT_SIGMA_0*SPLAT_SIGMA_0);
    float inv2_st = 1.0f/(2.0f*sigma_t*sigma_t);

    bool  aniso = (R > 1.0f);
    float rx = 0, ry = 0, rz = 0;
    if( aniso ) {
        rx = x/R;
        ry = y/R;
        rz = z/R;
    }

    /// t = dx + 5*dy + 25*dz, so lanes 0..4 share (dy,dz) and differ only in dx: five voxels
    /// adjacent in memory. 125 taps over 32 lanes is four rounds, the last one partial.
    for(int s=0; s<4; s++) {
        int t = lane + 32*s;
        if( t >= 125 ) break;

        int dx = (t     % 5) - SPLAT_RAD;
        int dy = ((t/5) % 5) - SPLAT_RAD;
        int dz = (t/25)      - SPLAT_RAD;

        int iz = iz0 + dz;
        int iy = iy0 + dy;
        int ix = ix0 + dx;

        if( (iz<-Nh) || (iz>=Nh) ) continue;
        if( (iy<-Nh) || (iy>=Nh) ) continue;

        float ex = float(ix) - x;
        float ey = float(iy) - y;
        float ez = float(iz) - z;

        float d2 = ex*ex + ey*ey + ez*ez;
        float e;
        if( aniso ) {
            float d_par = ex*rx + ey*ry + ez*rz;
            e = d_par*d_par*inv2_s0 + (d2 - d_par*d_par)*inv2_st;
        }
        else
            e = d2*inv2_s0;

        float g_wgt = __expf(-e)*inorm;
        if( g_wgt < SPLAT_W_MIN ) continue;
        g_wgt *= prj_w;

        /// Insert into hermitian volume
        bool should_conj = false;
        int ix_h = ix;
        int iy_h = iy;
        int iz_h = iz;

        if (ix_h < 0) {
            ix_h = -ix_h;
            iy_h = -iy_h;
            iz_h = -iz_h;
            should_conj = true;
        }
        iy_h += Nh;
        iz_h += Nh;

        if( (ix_h >= M) ) continue;
        if( (iy_h < 0) || (iy_h >= N) ) continue;
        if( (iz_h < 0) || (iz_h >= N) ) continue;

        long   idx = get_3d_idx(ix_h,iy_h,iz_h,M,N);
        float2 out = val;
        if( should_conj ) out.y = -out.y;

        atomic_Add( &(p_acc[idx].x) , g_wgt*out.x );
        atomic_Add( &(p_acc[idx].y) , g_wgt*out.y );
        atomic_Add( &(p_wgt[idx]  ) , fabsf(g_wgt)*ctf_wgt );
    }
}
#define KB_KERNEL_SUM 6.0024f
#define KB_W_MIN      (5e-3f*KB_KERNEL_SUM)

__global__ void insert_stk_kb_atomic(double2*p_acc,double*p_wgt,
                                    cudaTextureObject_t ss_stk, cudaTextureObject_t ss_wgt, const Proj2D*pTlt,
                                    const float3 bandpass,const int M, const int N, const int K)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x >= M || ss_idx.y >= N || ss_idx.z >= K )
        return;

    if( pTlt[ss_idx.z].w == 0 )
        return;

    int Nh = N/2;
    Vec3 pt;
    pt.x = ss_idx.x;
    pt.y = ss_idx.y - N/2;
    pt.z = 0;

    float R  = sqrtf( pt.x*pt.x + pt.y*pt.y );
    float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

    if( bp <= 1e-3f )
        return;

    float2 val     = tex2DLayered<float2>(ss_stk, float(ss_idx.x)+0.5, float(ss_idx.y)+0.5, ss_idx.z);
    float  ctf_wgt = tex2DLayered<float >(ss_wgt, float(ss_idx.x)+0.5, float(ss_idx.y)+0.5, ss_idx.z);
    float  prj_w   = pTlt[ss_idx.z].w * bp;

    float x,y,z;
    rot_pt(x,y,z,pTlt[ss_idx.z].R,pt);

    int ix0 = floorf(x);
    int iy0 = floorf(y);
    int iz0 = floorf(z);

    #pragma unroll
    for(int dz = -2; dz <= 2; dz++) {
        int iz = iz0 + dz;
        if( (iz<-Nh) || (iz>=Nh) ) continue;
        float tz = (z-iz)/2;
        if( fabsf(tz) > 1.0f  ) continue;
        float kbz = get_kaisser_bessel_kernel_polyfit(tz);

        #pragma unroll
        for(int dy = -2; dy <= 2; dy++) {
            int iy = iy0 + dy;
            if( (iy<-Nh) || (iy>=Nh) ) continue;
            float ty = (y-iy)/2;
            if( fabsf(ty) > 1.0f  ) continue;
            float kby = get_kaisser_bessel_kernel_polyfit(ty);

            #pragma unroll
            for(int dx = -2; dx <= 2; dx++) {
                int ix = ix0 + dx;
                float tx = (x-ix)/2;
                if( fabsf(tx) > 1.0f  ) continue;
                float kbx = get_kaisser_bessel_kernel_polyfit(tx);

                float kb_krn = kbx*kby*kbz;
                if( kb_krn < KB_W_MIN ) continue;

                /// Insert into hermitian volume
                bool should_conj = false;
                int ix_h = ix;
                int iy_h = iy;
                int iz_h = iz;

                if (ix_h < 0) {
                    ix_h = -ix_h;
                    iy_h = -iy_h;
                    iz_h = -iz_h;
                    should_conj = true;
                }
                iy_h += Nh;
                iz_h += Nh;

                if( (ix_h >=M) ) continue;
                if( (iy_h < 0) || (iy_h >= N) ) continue;
                if( (iz_h < 0) || (iz_h >= N) ) continue;

                float kb_wgt = kb_krn * prj_w;

                long   idx = get_3d_idx(ix_h,iy_h,iz_h,M,N);
                float2 out = val;
                if( should_conj ) out.y = -out.y;

                atomic_Add( &(p_acc[idx].x) , kb_wgt*out.x   );
                atomic_Add( &(p_acc[idx].y) , kb_wgt*out.y   );
                atomic_Add( &(p_wgt[idx]  ) , fabsf(kb_wgt)*ctf_wgt );
            }
        }
    }
}

/// --- Kaiser-Bessel gridding for slice extraction ---------------------------
///
/// Plain trilinear extraction is an interpolation whose low-pass strength
/// depends on the sub-voxel position of the sample, so grid-aligned orientations
/// come out sharper than off-grid ones and the angular search is biased toward
/// them. Widening the kernel to the same 4-voxel Kaiser-Bessel window already
/// used by insert_stk_kb_atomic removes that dependence.
///
/// Cost is kept at 8 fetches: the texture is sampled with cudaFilterModeLinear,
/// so one tex3D fetch already blends 2 voxels per axis. Pairing the window as
/// {i-1,i} and {i+1,i+2} and placing each fetch at the fractional position that
/// makes the hardware lerp reproduce the two KB weights is exact, turning 4x4x4
/// = 64 taps into 2x2x2 = 8 fetches.
///
/// The reference must be pre-divided by the kernel's transform before its FFT
/// (grid_correct_kb in upload_ref), otherwise the projection is of an apodised
/// reference and most of the benefit is lost.

/// Everything kb_lerp_pair needs depends only on frac(p), so it is tabulated
/// rather than recomputed. Entry i holds, for f = i/(KB_LUT_N-1):
///   x = c0 offset from floor(p),  y = a0,  z = c1 offset from floor(p),  w = a1
///
/// Evaluating the 8th-order polyfit four times and dividing four times per axis
/// costs 12 polynomial evaluations and 12 divisions per pixel, which measured at
/// 7.4x the cost of trilinear extraction with 98% of that in the arithmetic
/// (8 fetches alone are only 1.14x). The table brings it to 1.2x and is accuracy
/// neutral: max weight error 3e-7, identical CC spread to six decimals.
#define KB_LUT_N 1024
__device__ float4 g_kb_lut[KB_LUT_N];

namespace {
inline float kb_krn_host(const float t) {
    float t2=t*t, t4=t2*t2, t6=t4*t2, t8=t6*t2;
    float r = 0.99939224f;
    r -= 3.37246839f*t2;
    r += 4.70532537f*t4;
    r -= 3.26100335f*t6;
    r += 0.93508816f*t8;
    return r;
}
}

/// Builds and uploads g_kb_lut. Call once per device, after GPU::set_device and
/// before any extract_stk_kb launch.
inline void init_kb_lut() {
    float4 tbl[KB_LUT_N];
    for(int i=0;i<KB_LUT_N;i++) {
        float f  = (float)i/(float)(KB_LUT_N-1);
        /// kb_krn_host takes (offset/2); the offsets (-1-f,-f,1-f,2-f) all land
        /// inside the polyfit's valid |t|<=1 range.
        float w0 = fmaxf(kb_krn_host((-1.0f-f)*0.5f),0.0f);
        float w1 = fmaxf(kb_krn_host((     -f)*0.5f),0.0f);
        float w2 = fmaxf(kb_krn_host(( 1.0f-f)*0.5f),0.0f);
        float w3 = fmaxf(kb_krn_host(( 2.0f-f)*0.5f),0.0f);
        float s0 = fmaxf(w0+w1,1e-9f);
        float s1 = fmaxf(w2+w3,1e-9f);
        float s  = fmaxf(w0+w1+w2+w3,1e-6f);
        tbl[i] = make_float4( -1.0f + (w1/s0), s0/s, 1.0f + (w3/s1), s1/s );
    }
    cudaError_t err = cudaMemcpyToSymbol(g_kb_lut,tbl,sizeof(tbl));
    if( err != cudaSuccess ) {
        fprintf(stderr,"Error uploading the Kaiser-Bessel lookup table. ");
        fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
        exit(1);
    }
}

/// Collapses the 4-voxel window into two lerp fetch positions. a0+a1 == 1.
__device__ __forceinline__ void kb_lerp_pair(float&c0,float&a0,float&c1,float&a1,const float p) {
    float ip = floorf(p);
    float u  = (p - ip)*(KB_LUT_N-1);
    int   i  = min(max((int)u,0),KB_LUT_N-1);
    int   j  = min(i+1,KB_LUT_N-1);
    float t  = u - (float)i;
    float4 A = __ldg(&g_kb_lut[i]);
    float4 B = __ldg(&g_kb_lut[j]);
    c0 = ip + A.x + t*(B.x-A.x);
    a0 =      A.y + t*(B.y-A.y);
    c1 = ip + A.z + t*(B.z-A.z);
    a1 =      A.w + t*(B.w-A.w);
}

/// KB interpolation within one x column; ay/az must already sum to 1.
__device__ __forceinline__ float2 kb_plane(cudaTextureObject_t vol,const float cx,
                                           const float cy0,const float ay0,const float cy1,const float ay1,
                                           const float cz0,const float az0,const float cz1,const float az1) {
    float2 v00 = tex3D<float2>(vol,cx,cy0,cz0);
    float2 v10 = tex3D<float2>(vol,cx,cy1,cz0);
    float2 v01 = tex3D<float2>(vol,cx,cy0,cz1);
    float2 v11 = tex3D<float2>(vol,cx,cy1,cz1);
    float2 r;
    r.x = az0*(ay0*v00.x + ay1*v10.x) + az1*(ay0*v01.x + ay1*v11.x);
    r.y = az0*(ay0*v00.y + ay1*v10.y) + az1*(ay0*v01.y + ay1*v11.y);
    return r;
}

/// px is the (non-negative) kx of the sample, py/pz are centred frequencies.
///
/// The window reaches kx = -1 whenever px < 1, which is roughly 3.6% of the
/// slice pixels. That column is not stored in the Hermitian half-volume and
/// cudaAddressModeBorder would return zero for it, dropping up to 22% of the x
/// weight and costing about two orders of magnitude of accuracy. Those samples
/// take a second path that recovers it from
///     U(-1,ky,kz) = conj( U(1,-ky,-kz) ),
/// which is the same 4-tap interpolation of the kx = 1 plane evaluated at
/// (-ky,-kz) and conjugated: the KB window is even, so the weights carry over
/// unchanged. That path costs 12 fetches. Both paths are exact.
__device__ __forceinline__ float2 fetch_kb(cudaTextureObject_t vol,
                                           const float px,const float py,const float pz,const int N)
{
    const float Nh  = N/2;
    const float pxc = fmaxf(px,0.0f);

    float cy0,ay0,cy1,ay1;  kb_lerp_pair(cy0,ay0,cy1,ay1,py+Nh);
    float cz0,az0,cz1,az1;  kb_lerp_pair(cz0,az0,cz1,az1,pz+Nh);
    cy0 += 0.5f; cy1 += 0.5f; cz0 += 0.5f; cz1 += 0.5f;

    float2 acc;

    if( pxc >= 1.0f ) {
        float cx0,ax0,cx1,ax1;  kb_lerp_pair(cx0,ax0,cx1,ax1,pxc);
        float2 p0 = kb_plane(vol,cx0+0.5f,cy0,ay0,cy1,ay1,cz0,az0,cz1,az1);
        float2 p1 = kb_plane(vol,cx1+0.5f,cy0,ay0,cy1,ay1,cz0,az0,cz1,az1);
        acc.x = ax0*p0.x + ax1*p1.x;
        acc.y = ax0*p0.y + ax1*p1.y;
    }
    else {
        /// Window is kx = {-1,0,1,2}: kx=0 as a point fetch, {1,2} as a pair,
        /// and kx=-1 from the mirrored kx=1 plane.
        ///
        /// All four normalised weights come out of the same lerp pair. With
        /// ip = 0 here, t0 = c0+1 = w1/s0, so a0*t0 = w1/s, a0*(1-t0) = w0/s and
        /// a1 = (w2+w3)/s; c1 is already the fetch position for the {1,2} pair.
        float cx0,ax0,cx1,ax1;  kb_lerp_pair(cx0,ax0,cx1,ax1,pxc);
        float t0  = cx0 - floorf(pxc) + 1.0f;
        float w1n = ax0*t0;
        float w0n = ax0 - w1n;

        float2 p0 = kb_plane(vol,0.5f    ,cy0,ay0,cy1,ay1,cz0,az0,cz1,az1);
        float2 p1 = kb_plane(vol,cx1+0.5f,cy0,ay0,cy1,ay1,cz0,az0,cz1,az1);

        float my0,by0,my1,by1;  kb_lerp_pair(my0,by0,my1,by1,-py+Nh);
        float mz0,bz0,mz1,bz1;  kb_lerp_pair(mz0,bz0,mz1,bz1,-pz+Nh);
        float2 pm = kb_plane(vol,1.5f,my0+0.5f,by0,my1+0.5f,by1,mz0+0.5f,bz0,mz1+0.5f,bz1);

        acc.x = w1n*p0.x + ax1*p1.x + w0n*pm.x;
        acc.y = w1n*p0.y + ax1*p1.y - w0n*pm.y;   /// conj() on the mirror term
    }

    return acc;
}

__global__ void extract_stk_kb(float2*p_out,cudaTextureObject_t vol,const Proj2D*pTlt,
                               const float3 bandpass,const int M, const int N, const int K,
                               bool bandpass_squared=false)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < K ) {

        float2 val = {0,0};

        if( pTlt[ss_idx.z].w > 0 ) {
            Vec3 pt_in;
            pt_in.x = ss_idx.x;
            pt_in.y = ss_idx.y - N/2;
            pt_in.z = 0;

            float R = l2_distance(pt_in.x,pt_in.y);
            float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

            if( bp > 0.05 ) {
                if( bandpass_squared )
                    bp = bp*bp;
                Vec3 pt_out;
                rot_pt_XY(pt_out,pTlt[ss_idx.z].R,pt_in);

                bool should_conjugate = false;
                if( pt_out.x < 0 ) {
                    pt_out.x = -pt_out.x;
                    pt_out.y = -pt_out.y;
                    pt_out.z = -pt_out.z;
                    should_conjugate = true;
                }

                val = fetch_kb(vol,pt_out.x,pt_out.y,pt_out.z,N);
                val.x *= bp;
                val.y *= bp;

                if( should_conjugate )
                    val.y = -val.y;

            }
        }

        p_out[ ss_idx.x + M*ss_idx.y + M*N*ss_idx.z ] = val;
    }
}

__global__ void extract_stk_kb(float2*p_out,cudaTextureObject_t vol,const Proj2D*pTlt,
                               const Defocus*pDef,const float3 bandpass,
                               const int M, const int N, const int K,
                               bool bandpass_squared=false)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < K ) {

        float2 val = {0,0};

        if( pTlt[ss_idx.z].w > 0 ) {
            Vec3 pt_in;
            pt_in.x = ss_idx.x;
            pt_in.y = ss_idx.y - N/2;
            pt_in.z = 0;

            float max_R = bandpass.y;
            if( pDef[ss_idx.z].max_res > 0 )
                max_R = min(max_R,pDef[ss_idx.z].max_res);

            float R = l2_distance(pt_in.x,pt_in.y);
            float bp = get_bp_wgt(bandpass.x,max_R,bandpass.z,R);

            if( bp > 0.05 ) {
                if( bandpass_squared )
                    bp = bp*bp;
                Vec3 pt_out;
                rot_pt_XY(pt_out,pTlt[ss_idx.z].R,pt_in);

                bool should_conjugate = false;
                if( pt_out.x < 0 ) {
                    pt_out.x = -pt_out.x;
                    pt_out.y = -pt_out.y;
                    pt_out.z = -pt_out.z;
                    should_conjugate = true;
                }

                val = fetch_kb(vol,pt_out.x,pt_out.y,pt_out.z,N);
                val.x *= bp;
                val.y *= bp;

                if( should_conjugate )
                    val.y = -val.y;

            }
        }

        p_out[ ss_idx.x + M*ss_idx.y + M*N*ss_idx.z ] = val;
    }
}

__global__ void extract_stk(float2*p_out,cudaTextureObject_t vol,const Proj2D*pTlt,
                            const float3 bandpass,const int M, const int N, const int K,
                            bool bandpass_squared=false)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < K ) {

        float2 val = {0,0};

        if( pTlt[ss_idx.z].w > 0 ) {
            Vec3 pt_in;
            pt_in.x = ss_idx.x;
            pt_in.y = ss_idx.y - N/2;
            pt_in.z = 0;

            float R = l2_distance(pt_in.x,pt_in.y);
            float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

            if( bp > 0.05 ) {
                if( bandpass_squared )
                    bp = bp*bp;
                Vec3 pt_out;
                rot_pt_XY(pt_out,pTlt[ss_idx.z].R,pt_in);

                bool should_conjugate = false;
                if( pt_out.x < 0 ) {
                    pt_out.x = -pt_out.x;
                    pt_out.y = -pt_out.y;
                    pt_out.z = -pt_out.z;
                    should_conjugate = true;
                }

                val = tex3D<float2>(vol, pt_out.x+0.5, pt_out.y+N/2+0.5, pt_out.z+N/2+0.5);
                val.x *= bp;
                val.y *= bp;

                if( should_conjugate )
                    val.y = -val.y;

            }
        }

        p_out[ ss_idx.x + M*ss_idx.y + M*N*ss_idx.z ] = val;
    }
}

__global__ void extract_stk(float2*p_out,cudaTextureObject_t vol,const Proj2D*pTlt,
                            const Defocus*pDef,const float3 bandpass,
                            const int M, const int N, const int K,
                            bool bandpass_squared=false)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < K ) {

        float2 val = {0,0};

        if( pTlt[ss_idx.z].w > 0 ) {
            Vec3 pt_in;
            pt_in.x = ss_idx.x;
            pt_in.y = ss_idx.y - N/2;
            pt_in.z = 0;

            float max_R = bandpass.y;
            if( pDef[ss_idx.z].max_res > 0 )
                max_R = min(max_R,pDef[ss_idx.z].max_res);

            float R = l2_distance(pt_in.x,pt_in.y);
            float bp = get_bp_wgt(bandpass.x,max_R,bandpass.z,R);

            if( bp > 0.05 ) {
                if( bandpass_squared )
                    bp = bp*bp;
                Vec3 pt_out;
                rot_pt_XY(pt_out,pTlt[ss_idx.z].R,pt_in);

                bool should_conjugate = false;
                if( pt_out.x < 0 ) {
                    pt_out.x = -pt_out.x;
                    pt_out.y = -pt_out.y;
                    pt_out.z = -pt_out.z;
                    should_conjugate = true;
                }

                val = tex3D<float2>(vol, pt_out.x+0.5, pt_out.y+N/2+0.5, pt_out.z+N/2+0.5);
                val.x *= bp;
                val.y *= bp;

                if( should_conjugate )
                    val.y = -val.y;

            }
        }

        p_out[ ss_idx.x + M*ss_idx.y + M*N*ss_idx.z ] = val;
    }
}


__global__ void get_std_from_fourier(double*p_acc,cudaTextureObject_t vol,const float3 bandpass,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        Vec3 pt_in;
        pt_in.x = ss_idx.x;
        pt_in.y = ss_idx.y - ss_siz.y/2;
        pt_in.z = ss_idx.z - ss_siz.z/2;

        float R = l2_distance(pt_in.x,pt_in.y,pt_in.z);
        float bp = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

        if( (bp > 0.05) && (R > 0.5) ) {
            float2 val = tex3D<float2>(vol, pt_in.x+0.5, pt_in.y+ss_siz.y/2+0.5, pt_in.z+ss_siz.z/2+0.5);
            val.x *= bp;
            val.y *= bp;
            double acc = cuCabsf(val);
            atomic_Add( p_acc , acc );
        }
    }
}

__global__ void invert_wgt(double*p_data,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();
    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        long   idx  = get_3d_idx(ss_idx,ss_siz);
        double data = fmax(p_data[idx],1e-4);
        p_data[idx] = 1/data;
    }
}

__global__ void invert_wgt_arctan(double*p_data,float S, float F,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();
    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int center = ss_siz.y/2;
        float f = l2_distance(ss_idx.x,ss_idx.y-center,ss_idx.z-center);
        float  fsc    = 0.5 * ( 1 - (2/M_PI) * atan( (f-F)/S ) );
        float  lambda = (1-fsc)/fsc;
        long   idx    = get_3d_idx(ss_idx,ss_siz);
        double data   = p_data[idx] + lambda;
        p_data[idx] = 1/data;
    }
}

__global__ void invert_wgt_logistic(double*p_data,float S, float F,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();
    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int center = ss_siz.y/2;
        float f = l2_distance(ss_idx.x,ss_idx.y-center,ss_idx.z-center);
        float  fsc    = 1/(1+exp( (f-F)/S ));
        float  lambda = (1-fsc)/fsc;
        long   idx    = get_3d_idx(ss_idx,ss_siz);
        double data   = p_data[idx] + lambda;
        p_data[idx] = 1/data;
    }
}

__global__ void inv_wgt_ite_sphere(double*p_vol_wgt,const int3 ss_siz) {
    
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        int center = ss_siz.y/2;

        float R = l2_distance(ss_idx.x,ss_idx.y-center,ss_idx.z-center);

        double out = (R < center) ? 1.0 : 0.0;
        p_vol_wgt[ get_3d_idx(ss_idx,ss_siz) ] = out;
    }
}

/// Floors the sampling function before the Pipe & Menon iteration below. The floor has to apply
/// to exact zeros as well, and that is what bounds the whole iteration: where w is genuinely 0
/// the convolution stays 0 forever, so the divide hits its clamp every pass and the compensation
/// grows as clamp^-iterations without limit (1e20 after 10 passes at a 1e-2 clamp). With the
/// floor in place the convolution grows in step with the compensation, so the iteration settles
/// at ~1/(min_wgt*sum(C)) instead. Measured on a synthetic wedge: 1e20 -> 2.2e6, with an
/// identical fixed-point residual over the sampled voxels, so the bound costs no accuracy.
__global__ void inv_wgt_ite_hard_shrink(double*p_vol_wgt,double min_wgt,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        long idx = get_3d_idx(ss_idx,ss_siz);
        p_vol_wgt[ idx ] = fmax(p_vol_wgt[idx],min_wgt);
    }
}

__global__ void inv_wgt_ite_multiply(double*p_tmp,const double*p_vol_wgt,const double*p_wgt,const int3 ss_siz) {
    
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        long idx = get_3d_idx(ss_idx,ss_siz);
        double out = p_vol_wgt[idx]*p_wgt[idx];
        p_tmp[ idx ] = out;
    }
}

__global__ void inv_wgt_ite_convolve(double*p_conv,const double*p_tmp,const float4*p_krnl,const int n_krnl,const int3 ss_siz) {
    
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        double out = 0;

        for(int i=0; i<n_krnl; i++ ) {
            int x = ss_idx.x + (int)(p_krnl[i].x);
            int y = ss_idx.y + (int)(p_krnl[i].y);
            int z = ss_idx.z + (int)(p_krnl[i].z);
            double w = p_krnl[i].w;
            
            if( get_mirror_index(x,y,z,ss_siz.x,ss_siz.y) ) {
                out += w*p_tmp[ get_3d_idx(x,y,z,ss_siz) ];
            }
        }

        p_conv[ get_3d_idx(ss_idx,ss_siz) ] = out;
    }
}

/// The clamp is only a backstop now: inv_wgt_ite_hard_shrink guarantees the convolution is
/// strictly positive, so it is no longer what bounds the iteration. Measured with that floor in
/// place, anything from 1e-2 to 1e-8 settles at the same magnitude and the same residual, so
/// this matches the value RELION uses rather than the looser 1e-2 that used to carry the bound.
#define INV_WGT_ITE_MIN_DEN 1e-6

__global__ void inv_wgt_ite_divide(double*p_vol_wgt, const double*p_conv,const int3 ss_siz) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        long idx = get_3d_idx(ss_idx,ss_siz);
        double den = p_conv[idx];
        den = copysign(fmax(fabs(den),INV_WGT_ITE_MIN_DEN),den);
        p_vol_wgt[ idx ] = p_vol_wgt[ idx ] / den;
    }
}

__global__ void grid_correct_linear(float*p_data,const int N) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < N && ss_idx.y < N && ss_idx.z < N ) {

        long ix = ss_idx.x + ss_idx.y*N + ss_idx.z*N*N;

        int center = N/2;
        float tx = ss_idx.x-center;
        float ty = ss_idx.y-center;
        float tz = ss_idx.z-center;

        float wx = sinc(tx/N);
        float wy = sinc(ty/N);
        float wz = sinc(tz/N);

        float wgt = wx*wx*wy*wy*wz*wz;
        wgt = fminf( fmaxf(wgt,1e-5f), 1-1e-5f );
        float val = p_data[ix];
        p_data[ix] = val/wgt;
    }
}

__global__ void grid_correct_kb(float*p_data,const int N) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < N && ss_idx.y < N && ss_idx.z < N ) {

        long ix = ss_idx.x + ss_idx.y*N + ss_idx.z*N*N;

        int center = N/2;
        float tx = ss_idx.x-center;
        float ty = ss_idx.y-center;
        float tz = ss_idx.z-center;
        
        const float kb_dc = get_kaisser_bessel_correction_polyfit(0.0f);

        float kbx = get_kaisser_bessel_correction_polyfit(tx/center)/kb_dc;
        float kby = get_kaisser_bessel_correction_polyfit(ty/center)/kb_dc;
        float kbz = get_kaisser_bessel_correction_polyfit(tz/center)/kb_dc;

        float wgt = kbx*kby*kbz;
        wgt = fmaxf(wgt,1e-5f);
        float val = p_data[ix];
        p_data[ix] = val/wgt;
    }
}

#define GRID_CORRECT_FWD_MIN 1e-2f

__global__ void grid_correct_kb_fwd(float*p_data,const int N) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < N && ss_idx.y < N && ss_idx.z < N ) {

        long ix = ss_idx.x + ss_idx.y*N + ss_idx.z*N*N;

        int center = N/2;
        float tx = ss_idx.x-center;
        float ty = ss_idx.y-center;
        float tz = ss_idx.z-center;

        float kbx = get_kb_fwd_correction_polyfit(tx/center);
        float kby = get_kb_fwd_correction_polyfit(ty/center);
        float kbz = get_kb_fwd_correction_polyfit(tz/center);

        float wgt = kbx*kby*kbz;
        wgt = fminf( fmaxf(wgt,GRID_CORRECT_FWD_MIN), 1.0f );
        float val = p_data[ix];
        p_data[ix] = val/wgt;
    }
}

__global__ void boost_low_freq(float2*p_out,
                               const float scale, const float value, const float decay,
                               const int3 ss_siz)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {

        long ix = ss_idx.x + ss_idx.y*ss_siz.x + ss_idx.z*ss_siz.x*ss_siz.y;
        int center = ss_siz.y/2;
        float R = l2_distance(ss_idx.x,ss_idx.y-center,ss_idx.z-center);

        float2 val = p_out[ix];

        float bp = get_bp_wgt(0,value,decay,R);
        bp = ((scale*bp)+1)/(scale+1);
        val.x *= bp;
        val.y *= bp;

        p_out[ ix ] = val;
    }
}

__global__ void add_symmetry(double2*p_val,double*p_wgt,
                             const double2*t_val, const double*t_wgt,
                             Rot33 Rsym, const int M, const int N)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < N ) {

        Vec3 pt;
        pt.x = ss_idx.x;
        pt.y = ss_idx.y - N/2;
        pt.z = ss_idx.z - N/2;

        float R = sqrt( pt.x*pt.x + pt.y*pt.y + pt.z*pt.z );

        if( R < (N/2) ) {

            long idx = ss_idx.x + ss_idx.y*M + ss_idx.z*M*N;
            double2 val = p_val[idx];
            double  wgt = p_wgt[idx];
            float x,y,z;

            bool should_conj = false;
            bool should_read = false;

            rot_inv_pt(x,y,z,Rsym,pt);

            int p_x = floor(x);
            int p_y = floor(y);
            int p_z = floor(z);
            float w_x = x - floor(x);
            float w_y = y - floor(y);
            float w_z = z - floor(z);

            int     read_idx;
            double2 read_val;
            double  read_wgt;
            float   w;

            read_idx = get_ix_3d(should_read,should_conj,p_x  ,p_y  ,p_z  ,M,N);
            read_val = t_val[read_idx];
            read_wgt = t_wgt[read_idx];
            if(should_conj) read_val.y = -read_val.y;
            w = (1-w_x)*(1-w_y)*(1-w_z);
            val.x += w*read_val.x;
            val.y += w*read_val.y;
            wgt   += w*read_wgt;

            read_idx = get_ix_3d(should_read,should_conj,p_x+1,p_y  ,p_z  ,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (  w_x)*(1-w_y)*(1-w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            read_idx = get_ix_3d(should_read,should_conj,p_x  ,p_y+1,p_z  ,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (1-w_x)*(  w_y)*(1-w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            read_idx = get_ix_3d(should_read,should_conj,p_x+1,p_y+1,p_z  ,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (  w_x)*(  w_y)*(1-w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            read_idx = get_ix_3d(should_read,should_conj,p_x  ,p_y  ,p_z+1,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (1-w_x)*(1-w_y)*(  w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            read_idx = get_ix_3d(should_read,should_conj,p_x+1,p_y  ,p_z+1,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (  w_x)*(1-w_y)*(  w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            read_idx = get_ix_3d(should_read,should_conj,p_x  ,p_y+1,p_z+1,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (1-w_x)*(  w_y)*(  w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            read_idx = get_ix_3d(should_read,should_conj,p_x+1,p_y+1,p_z+1,M,N);
            if( should_read ) {
                read_val = t_val[read_idx];
                read_wgt = t_wgt[read_idx];
                if(should_conj) read_val.y = -read_val.y;
                w = (  w_x)*(  w_y)*(  w_z);
                val.x += w*read_val.x;
                val.y += w*read_val.y;
                wgt   += w*read_wgt;
            }

            p_val[idx] = val;
            p_wgt[idx] = wgt;
        }
    }

}

__global__ void reconstruct_pts(float*p_cc,const Proj2D*pTlt,cudaTextureObject_t ss_cc,
                                const Rot33 R,const Vec3*p_pts,const int n_pts,
                                const int N,const int K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < n_pts && ss_idx.y < 1 && ss_idx.z < 1 ) {

        float cc  = 0;
        float wgt = 0;
        Vec3  pt = p_pts[ss_idx.x];
        float rx,ry,rz;
        rot_inv_pt(rx,ry,rz,R,pt);
        Vec3  pt_r = {rx,ry,rz};
        single x,y;
        single off = (single)(N/2) + 0.5;

        for(int z=0;z<K;z++) {
            if( pTlt[z].w > SUSAN_FLOAT_TOL  ) {
                rot_inv_pt_XY(x,y,pTlt[z].R,pt_r);
                cc  += pTlt[z].w*tex2DLayered<float>(ss_cc,x+off,y+off,z);
                wgt += pTlt[z].w;
            }
        }
        
        if( wgt == 0 ) wgt = 1;

        p_cc[ss_idx.x] = cc/wgt;
    }

}

__global__ void extract_pts(float*p_cc,const float*p_data,const Proj2D*pTlt,const Vec3*p_pts,const int n_pts,const int N,const int K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < n_pts && ss_idx.y < 1 && ss_idx.z < K ) {

        Vec3  pt = p_pts[ss_idx.x];

        int x = (int)pt.x + N/2;
        int y = (int)pt.y + N/2;

        float cc = p_data[x + y*N + ss_idx.z*N*N];

        p_cc[ss_idx.x + n_pts*ss_idx.z] = pTlt[ss_idx.z].w*cc;

    }

}

__global__ void extract_pts(float*p_cc,cudaTextureObject_t ss_cc,const Proj2D*pTlt,const Vec3*p_pts,const int n_pts,const int N,const int K) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < n_pts && ss_idx.y < 1 && ss_idx.z < K ) {

        Vec3  pt = p_pts[ss_idx.x];

        single x = (single)((int)pt.x + N/2) + 0.5;
        single y = (single)((int)pt.y + N/2) + 0.5;

        float cc = tex2DLayered<float>(ss_cc,x,y,ss_idx.z);

        p_cc[ss_idx.x + n_pts*ss_idx.z] = pTlt[ss_idx.z].w*cc;

    }

}

__global__ void multiply_vol2(float2*p_out,cudaTextureObject_t vol_tex,const float2*p_data,
                              const Rot33 Rot,const float3 bandpass,const int M, const int N,const float den=1)
{
    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < N ) {

        long idx = ss_idx.x + ss_idx.y*M + ss_idx.z*M*N;

        float2 val = {0,0};

        Vec3 pt_in;
        pt_in.x = ss_idx.x;
        pt_in.y = ss_idx.y - N/2;
        pt_in.z = ss_idx.z - N/2;

        float R = l2_distance(pt_in.x,pt_in.y,pt_in.z);
        float w = get_bp_wgt(bandpass.x,bandpass.y,bandpass.z,R);

        if( w > 0.05 ) {
            w = w/den;
            float x,y,z;
            rot_pt(x,y,z,Rot,pt_in);

            bool should_conj=false;
            if( x < 0 ) {
                x = -x;
                y = -y;
                z = -z;
                should_conj=true;
            }

            float2 val_a = tex3D<float2>(vol_tex,x+0.5,y+N/2+0.5,z+N/2+0.5);

            if( should_conj )
                val_a.y = -val_a.y;

            float2 val_b = p_data[idx];

            val = cuCmulf(val_a,val_b);
            val.x *= w;
            val.y *= w;
        }

        p_out[ idx ] = val;
    }
}

__global__ void extract_pts(float*p_cc,const float*p_data,const Vec3*p_pts,const int n_pts,const int N) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < n_pts && ss_idx.y < 1 && ss_idx.z < 1 ) {

        Vec3  pt = p_pts[ss_idx.x];

        int x = (int)pt.x + N/2;
        int y = (int)pt.y + N/2;
        int z = (int)pt.z + N/2;

        float cc = p_data[x + y*N + z*N*N];

        p_cc[ss_idx.x] = cc;

    }

}

__global__ void radial_cc(float*p_acc,const float2*p_vol_a,const float2*p_vol_b,const int M,const int N,const float scale=1) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < N ) {

        int center = N/2;
        float R = l2_distance(ss_idx.x,ss_idx.y-center,ss_idx.z-center);
        int r = round(R);

        if( r<M ) {
            long ix = ss_idx.x + ss_idx.y*M + ss_idx.z*M*N;
            float2 v_a = p_vol_a[ix];
            float2 v_b = p_vol_b[ix];
            v_b.y = -v_b.y;
            float2 val = cuCmulf(v_a,v_b);
            val.x = val.x*scale;
            atomicAdd(p_acc+r,val.x);
        }
    }
}

__global__ void calc_fsc(float*p_fsc,const float*p_den_a,const float*p_den_b,const int M) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < 1 && ss_idx.z < 1 ) {

        float num = p_fsc[ss_idx.x];
        float den = p_den_a[ss_idx.x]*p_den_b[ss_idx.x];
        if(den<0)
            den =1;
        den = sqrt(den);
        if(den<1e-9 && abs(num)<1e-9) {
            num = 1;
            den = 1;
        }

        p_fsc[ss_idx.x] = num/den;
    }
}

__global__ void randomize_phase(float2*p_vol,const float*p_ang,const float fpix,const int M,const int N) {

    int3 ss_idx = get_th_idx();

    if( ss_idx.x < M && ss_idx.y < N && ss_idx.z < N ) {

        long ix = ss_idx.x + ss_idx.y*M + ss_idx.z*M*N;
        float2 vol = p_vol[ix];

        int center = N/2;
        float R = l2_distance(ss_idx.x,ss_idx.y-center,ss_idx.z-center);
        int r = round(R);

        if( r>=fpix ) {
            float ang = p_ang[ix];
            float2 v_a = vol;
            float2 v_b;
            v_b.x = cos(ang);
            v_b.y = sin(ang);
            vol = cuCmulf(v_b,v_a);
        }
        p_vol[ix]=vol;
    }
}


}

#endif /// GPU_KERNEL_VOL_H



