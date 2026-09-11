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

#ifndef REFS_ALIGNER_H
#define REFS_ALIGNER_H

#include <iostream>
#include "datatypes.h"
#include "particles.h"
#include "reference.h"
#include "ref_maps.h"
#include "gpu.h"
#include "gpu_fft.h"
#include "gpu_kernel.h"
#include "mrc.h"
#include "io.h"
#include "points_provider.h"
#include "angles_provider.h"
#include "ref_ali.h"
#include "refs_aligner_args.h"

class RefsAligner {

public:
    ArgsRefsAli::Info *p_info;
    RefMap *p_refs;
    int N;
    int M;
    int R;

    M33f*Rali;
    V3f *Tali;

    Vec3 *c_pts;
    uint32 n_pts;
    GPU::GArrVec3 g_pts;

    float *c_cc;
    GPU::GArrSingle g_cc;

    GPU::GArrSingle  real_buffer;
    GPU::GArrSingle2 vol_b;
    GPU::GArrSingle2 vol_cc;
    GPU::GTex3DSingle2 vol_a;

    GpuFFT::FFT3D  fft3;
    GpuFFT::IFFT3D ifft3;

    AnglesProvider ang_prov;

    float3 bandpass;

    RefsAligner(ArgsRefsAli::Info*info,References*in_p_refs) {
        p_info = info;
        N = info->box_size;
        M = (N/2)+1;
        load_references(in_p_refs);
        setup_gpu();
        create_pts();

        ang_prov.cone_range = p_info->cone_range;
        ang_prov.cone_step  = p_info->cone_step;
        ang_prov.inplane_range = p_info->inplane_range;
        ang_prov.inplane_step  = p_info->inplane_step;
        ang_prov.refine_factor = p_info->refine_factor;
        ang_prov.refine_level  = p_info->refine_level;

        float bp_pad = p_info->fpix_roll/2;
        bandpass.x = max(p_info->fpix_min-bp_pad,0.0);
        bandpass.y = min(p_info->fpix_max+bp_pad,(float)N);
        bandpass.z = sqrt(p_info->fpix_roll);
    }

    ~RefsAligner() {
        delete [] p_refs;
        delete [] Rali;
        delete [] Tali;
        delete [] c_cc;
        delete [] c_pts;
    }

    void align() {
        V3f eu;
        for(int r=0;r<R;r++) {
            printf("\tAligning reference %2d...",r+1); fflush(stdout);
            align_half_maps(Rali[r],Tali[r],p_refs[r]);
            Math::Rmat_eZYZ(eu,Rali[r]);
            eu *= RAD2DEG;
            printf(" R=[%7.2f %7.2f %7.2f] t=[%4.1f %4.1f %4.1f] Done.\n",eu(0),eu(1),eu(2),Tali[r](0),Tali[r](1),Tali[r](2)); fflush(stdout);
        }
    }

    void update_ptcls(Particles&ptcls) {
        Particle ptcl;
        M33f Rold,Rnew;
        V3f eu,delta;
        int r;
        for(int i=0;i<ptcls.n_ptcl;i++) {
            ptcls.get(ptcl,i);
            if( ptcl.half_id() == 2 ) {
                r = ptcl.ref_cix();
                eu(0) = ptcl.ali_eu[r].x;
                eu(1) = ptcl.ali_eu[r].y;
                eu(2) = ptcl.ali_eu[r].z;
                Math::eZYZ_Rmat(Rold,eu);
                Rnew = Rali[r]*Rold;
                Math::Rmat_eZYZ(eu,Rnew);
                ptcl.ali_eu[r].x = eu(0);
                ptcl.ali_eu[r].y = eu(1);
                ptcl.ali_eu[r].z = eu(2);
                delta = p_info->pix_size*Rnew.transpose()*Tali[r];
                ptcl.ali_t[r].x += delta(0);
                ptcl.ali_t[r].y += delta(1);
                ptcl.ali_t[r].z += delta(2);
            }
        }
    }

protected:
    void load_references(References*in_p_refs) {
        R = in_p_refs->num_refs;
        p_refs = new RefMap[R];
        for(int r=0;r<R;r++) {
            p_refs[r].load(in_p_refs->at(r));
        }
        Rali = new M33f[R];
        Tali = new V3f [R];
        for(int r=0;r<R;r++) {
            Rali[r] = M33f::Identity();
            Tali[r] = V3f::Zero();
        }
    }

    void setup_gpu() {
        GPU::set_device(p_info->gpu_ix);
        real_buffer.alloc(N*N*N);
        vol_a.alloc(M,N,N);
        vol_b.alloc(M*N*N);
        vol_cc.alloc(M*N*N);
        fft3.alloc(N);
        ifft3.alloc(N);
    }

    void create_pts() {
        if( p_info->off_type == ELLIPSOID )
            c_pts = PointsProvider::ellipsoid(n_pts,p_info->off_x,p_info->off_y,p_info->off_z,p_info->off_s);
        if( p_info->off_type == CYLINDER )
            c_pts = PointsProvider::cylinder(n_pts,p_info->off_x,p_info->off_y,p_info->off_z,p_info->off_s);

        c_cc = new float[n_pts];
        g_cc.alloc(n_pts);
        g_pts.alloc(n_pts);
        cudaMemcpy((void*)g_pts.ptr,(const void*)c_pts,sizeof(Vec3)*n_pts,cudaMemcpyHostToDevice);
    }

    void align_half_maps(M33f&R_ref,V3f&t_ref,RefMap&ref) {

        load_map(vol_b,ref.half_A);
        load_surf(vol_a,vol_b);
        load_map(vol_b,ref.half_B);

        GPU::sync();

        Rot33 Rot;
        single max_cc=-INFINITY,cc;
        int max_idx=0,idx;
        M33f R_ite,R_tmp;
        M33f R_lvl = Eigen::MatrixXf::Identity(3,3);

        R_ref = M33f::Identity();

        for( ang_prov.levels_init(); ang_prov.levels_available(); ang_prov.levels_next() ) {
            for( ang_prov.sym_init(); ang_prov.sym_available(); ang_prov.sym_next() ) {
                for( ang_prov.cone_init(); ang_prov.cone_available(); ang_prov.cone_next() ) {
                    for( ang_prov.inplane_init(); ang_prov.inplane_available(); ang_prov.inplane_next() ) {

                        ang_prov.get_current_R(R_ite);
                        R_tmp = R_ite*R_lvl;
                        Math::set(Rot,R_tmp);

                        cross_correlate(cc,idx,Rot,vol_a,vol_b);

                        if( cc > max_cc ) {
                            max_idx = idx;
                            max_cc  = cc;
                            R_ref   = R_tmp;
                        }
                    } // INPLANE
                } // CONE
            } // SYMMETRY
            R_lvl = R_ref;
        } // REFINE

        t_ref(0) = c_pts[max_idx].x;
        t_ref(1) = c_pts[max_idx].y;
        t_ref(2) = c_pts[max_idx].z;
    }

    void load_map(GPU::GArrSingle2&vol_out,const float*vol_in) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grdR = GPU::calc_grid_size(blk,N,N,N);
        dim3 grdC = GPU::calc_grid_size(blk,M,N,N);
        cudaMemcpy((void*)real_buffer.ptr,(const void*)vol_in,sizeof(single)*N*N*N,cudaMemcpyHostToDevice);
        GpuKernels::fftshift3D<<<grdR,blk>>>(real_buffer.ptr,N);
        fft3.exec(vol_out.ptr,real_buffer.ptr);
        GpuKernels::fftshift3D<<<grdC,blk>>>(vol_out.ptr,M,N);
    }

    void load_surf(GPU::GTex3DSingle2&vol_out,GPU::GArrSingle2&vol_in) {
        bool should_conjugate = true;
        int3 siz = make_int3(M,N,N);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,M,N,N);
        GpuKernels::load_surf_3<<<grd,blk>>>(vol_out.surface,vol_in.ptr,siz,should_conjugate);
    }

    void cross_correlate(float&cc,int&idx,Rot33&Rot,GPU::GTex3DSingle2&vol_in_a,GPU::GArrSingle2&vol_in_b) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grdR = GPU::calc_grid_size(blk,N,N,N);
        dim3 grdC = GPU::calc_grid_size(blk,M,N,N);
        float den = N*N*N;
        GpuKernelsVol::multiply_vol2<<<grdC,blk>>>(vol_cc.ptr,vol_in_a.texture,vol_in_b.ptr,Rot,bandpass,M,N,den*den);
        GpuKernels::fftshift3D<<<grdC,blk>>>(vol_cc.ptr,M,N);
        ifft3.exec(real_buffer.ptr,vol_cc.ptr);
        GpuKernels::fftshift3D<<<grdR,blk>>>(real_buffer.ptr,N);

        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grdR.x = GPU::div_round_up(n_pts,1024);
        grdR.y = 1;
        grdR.z = 1;
        GpuKernelsVol::extract_pts<<<grdR,blk>>>(g_cc.ptr,real_buffer.ptr,g_pts.ptr,n_pts,N);
        cudaMemcpy((void*)c_cc,(const void*)g_cc.ptr,sizeof(single)*n_pts,cudaMemcpyDeviceToHost);
        cc = 0;
        for(int i=0;i<n_pts;i++) {
            if( cc < c_cc[i] ) {
                cc = c_cc[i];
                idx = i;
            }
        }
    }
};

#endif /// REFS_ALIGNER_H


