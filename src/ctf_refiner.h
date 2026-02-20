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

#ifndef CTF_REFINER_H
#define CTF_REFINER_H

#include <iostream>
#include "datatypes.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "pool_coordinator.h"
#include "particles.h"
#include "tomogram.h"
#include "reference.h"
#include "ref_maps.h"
#include "stack_reader.h"
#include "gpu.h"
#include "gpu_kernel.h"
#include "gpu_rand.h"
#include "gpu_kernel_ctf.h"
#include "substack_crop.h"
#include "mrc.h"
#include "io.h"
#include "ref_ali.h"
#include "ctf_refiner_args.h"
#include "progress.h"

typedef enum {
    CTF_REF=1
} CtfRefCmd;

class CtfRefBuffer {

public:
    GPU::GHostSingle  c_stk;
    GPU::GHostFloat2  c_pad;
    GPU::GHostProj2D  c_ali;
    GPU::GHostDefocus c_def;
    GPU::GArrSingle   g_stk;
    GPU::GArrSingle2  g_pad;
    GPU::GArrProj2D   g_ali;
    GPU::GArrDefocus  g_def;
    Particle ptcl;
    CtfConst ctf_vals;
    int      K;
    int      r_ix;
    int      class_ix;

    CtfRefBuffer(int N,int max_k) {
        c_stk.alloc(N*N*max_k);
        g_stk.alloc(N*N*max_k);
        c_pad.alloc(max_k);
        g_pad.alloc(max_k);
        c_ali.alloc(max_k);
        g_ali.alloc(max_k);
        c_def.alloc(max_k);
        g_def.alloc(max_k);
        K = 0;
    }

    ~CtfRefBuffer() {
    }

};

class CtfRefGpuWorker : public Worker {

protected:
    class Refine{
    public:
        int M;
        int N;
        int max_K;

        GPU::GArrSingle2 prj_buf;
        GPU::GArrSingle  ctf_ite;

        Refine(int m, int n, int k) {
            M = m;
            N = n;
            max_K = k;
            prj_buf.alloc(m*n*k);
            ctf_ite.alloc(m*n*k);
        }

        ~Refine() {
        }

        void load_buf(GPU::GArrSingle2&data_in,int k,GPU::Stream&stream) {
            GPU::copy_async(prj_buf.ptr,data_in.ptr,M*N*k,stream.strm);
        }

        void apply_ctf(GPU::GArrSingle2&data_out,float3 delta,RadialAverager&rad_avgr,CtfRefBuffer*ptr,GPU::Stream&stream) {
            int3 ss = make_int3(M,N,ptr->K);
            dim3 blk = GPU::get_block_size_2D();
            dim3 grd = GPU::calc_grid_size(blk,M,N,ptr->K);
            GpuKernelsCtf::create_ctf<<<grd,blk,0,stream.strm>>>(ctf_ite.ptr,delta,ptr->ctf_vals,ptr->g_def.ptr,false,ss);
            rad_avgr.normalize_ctf(ctf_ite,ptr->K,stream);
            GpuKernels::multiply<<<grd,blk,0,stream.strm>>>(data_out.ptr,prj_buf.ptr,ctf_ite.ptr,ss);
        }
    };

public:
    int gpu_ix;
    int N;
    int M;
    int P;
    int R;
    int pad_type;
    int max_K;
    float def_range;
    float def_step;
    float ang_range;
    float ang_step;
    bool  est_dose;
    bool  use_halves;
    bool  astigmatism;
    float3 bandpass;
    float2 ssnr; /// x=F; y=S;
    float4 off_par;

    DoubleBufferHandler *p_buffer;
    RefMap              *p_refs;

    CtfRefGpuWorker() {
    }

    ~CtfRefGpuWorker() {
    }

protected:
    int NP;
    int MP;

    void main() {

        NP = N+P;
        MP = (NP/2)+1;

        GPU::set_device(gpu_ix);
        int current_cmd;
        GPU::Stream stream;
        stream.configure();

        AliSubstack ss_data(M,N,max_K,P,stream);

        uint32 off_type = CIRCLE;

        AliData ali_data(MP,NP,max_K,off_par,off_type,stream);

        RadialAverager rad_avgr(M,N,max_K);

        Refine ctf_ref(MP,NP,max_K);

        int num_vols = R;
        if( use_halves ) num_vols = 2*R;
        AliRef*vols = new AliRef[num_vols];
        allocate_references(vols,rad_avgr);

        GPU::sync();

        while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
            switch(current_cmd) {
                case CTF_REF:
                    refine_ctf(vols,ctf_ref,ss_data,ali_data,rad_avgr,stream);
                    break;
                default:
                    break;
            }
        }

        GPU::sync();
        delete [] vols;
    }

    void allocate_references(AliRef*vols,RadialAverager&rad_avgr) {
        GPU::GArrSingle  g_raw;
        GPU::GArrSingle  g_pad;
        GPU::GArrSingle2 g_fou;

        g_raw.alloc(N*N*N);
        g_pad.alloc(NP*NP*NP);
        g_fou.alloc(MP*NP*NP);

        GpuFFT::FFT3D fft3;
        fft3.alloc(NP);

        if( use_halves ) {
            for(int r=0;r<R;r++) {
                upload_ref(g_pad,g_raw,p_refs[r].half_A);
                exec_fft3(g_fou,g_pad,fft3);
                vols[2*r  ].allocate(g_fou,MP,NP);

                upload_ref(g_pad,g_raw,p_refs[r].half_B);
                exec_fft3(g_fou,g_pad,fft3);
                vols[2*r+1].allocate(g_fou,MP,NP);
            }
        }
        else {
            for(int r=0;r<R;r++) {
                upload_ref(g_pad,g_raw,p_refs[r].map);
                exec_fft3(g_fou,g_pad,fft3);
                vols[r].allocate(g_fou,MP,NP);
            }
        }
    }

    void upload_ref(GPU::GArrSingle&g_pad,GPU::GArrSingle&g_raw,single*data) {
        cudaError_t err = cudaMemcpy((void*)g_raw.ptr,(const void*)data,sizeof(single)*N*N*N,cudaMemcpyHostToDevice);
        if( err != cudaSuccess ) {
            fprintf(stderr,"Error uploading volume to CUDA memory. ");
            fprintf(stderr,"GPU error: %s.\n",cudaGetErrorString(err));
            exit(1);
        }
        g_pad.clear();

        int3 pad = make_int3(P/2,P/2,P/2);
        int3 ss_raw = make_int3(N,N,N);
        int3 ss_pad = make_int3(NP,NP,NP);

        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,N,N,N);

        GpuKernels::load_pad<<<grd,blk>>>(g_pad.ptr,g_raw.ptr,pad,ss_raw,ss_pad);
    }

    void exec_fft3(GPU::GArrSingle2&g_fou,GPU::GArrSingle&g_pad,GpuFFT::FFT3D&fft3) {
        dim3 blk = GPU::get_block_size_2D();
        dim3 grdR = GPU::calc_grid_size(blk,NP,NP,NP);
        dim3 grdC = GPU::calc_grid_size(blk,MP,NP,NP);
        GpuKernels::fftshift3D<<<grdR,blk>>>(g_pad.ptr,NP);
        GpuKernels::sampling_correction_3D<<<grdC,blk>>>(g_fou.ptr,2.0,MP,NP);
        fft3.exec(g_fou.ptr,g_pad.ptr);
        GpuKernels::fftshift3D<<<grdC,blk>>>(g_fou.ptr,MP,NP);
    }

    void refine_ctf(AliRef*vols,Refine&ctf_ref,AliSubstack&ss_data,AliData&ali_data,RadialAverager&rad_avgr,GPU::Stream&stream) {
        p_buffer->RO_sync();
        while( p_buffer->RO_get_status() > DONE ) {
            if( p_buffer->RO_get_status() == READY ) {
                CtfRefBuffer*ptr = (CtfRefBuffer*)p_buffer->RO_get_buffer();
                add_data(ss_data,ptr,rad_avgr,stream);
                search_ctf(vols[ptr->r_ix],ss_data,ctf_ref,ptr,ali_data,rad_avgr,stream);
                stream.sync();
            }
            p_buffer->RO_sync();
        }
    }

    void add_data(AliSubstack&ss_data,CtfRefBuffer*ptr,RadialAverager&rad_avgr,GPU::Stream&stream) {

        switch( pad_type ) {
            case PAD_ZERO:
                ss_data.pad_zero(stream);
                break;
            case PAD_GAUSSIAN:
                ss_data.pad_normal(ptr->g_pad,ptr->K,stream);
                break;
            default:
                break;
        }

        ss_data.add_data(ptr->g_stk,ptr->g_ali,ptr->K,stream);

        rad_avgr.calculate_FRC(ss_data.ss_fourier,ptr->K,stream);
        rad_avgr.apply_FRC(ss_data.ss_fourier,ptr->ctf_vals,ssnr,ptr->K,stream);
    }

    void debug_fourier_stack(const char*filename,GPU::GArrSingle2&g_fou,GPU::Stream&stream) {
        GPU::GArrSingle2 g_work;
        g_work.alloc( NP*NP*max_K );
        GPU::copy_async(g_work.ptr,g_fou.ptr,MP*NP*max_K,stream.strm);

        GpuFFT::IFFT2D ifft2;
        ifft2.alloc(MP,NP,max_K);
        ifft2.set_stream(stream.strm);

        GPU::GArrSingle g_real;
        g_real.alloc( NP*NP*max_K );

        GPU::GHostSingle buffer;
        buffer.alloc(NP*NP*max_K);

        int3 ss_fou = make_int3(MP,NP,max_K);
        int3 ss_pad = make_int3(NP,NP,max_K);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd_f = GPU::calc_grid_size(blk,MP,NP,max_K);
        dim3 grd_r = GPU::calc_grid_size(blk,NP,NP,max_K);

        GpuKernels::fftshift2D<<<grd_f,blk,0,stream.strm>>>(g_work.ptr,ss_fou);
        GpuKernels::sampling_correction_2D<<<grd_f,blk,0,stream.strm>>>(g_work.ptr,0.5,ss_fou);
        ifft2.exec(g_real.ptr,g_work.ptr);
        GpuKernels::fftshift2D<<<grd_r,blk,0,stream.strm>>>(g_real.ptr,ss_pad);
        GPU::download_async(buffer.ptr,g_real.ptr,NP*NP*max_K,stream.strm);
        stream.sync();

        Mrc::write(buffer.ptr,NP,NP,max_K,filename);
    }

    void search_ctf(AliRef&vol,AliSubstack&ss_data,Refine&ctf_ref,CtfRefBuffer*ptr,AliData&ali_data,RadialAverager&rad_avgr,GPU::Stream&stream) {

        Rot33 Rot;
        single max_cc[ptr->K];
        single sum_cc[ptr->K];
        single ite_cc[ptr->K];
        int    max_idx[ptr->K];
        int    ite_idx[ptr->K];
        Defocus def_rslt[ptr->K];

        for(int i=0;i<ptr->K;i++) max_cc[i] = -INFINITY;
        memset(def_rslt,0,sizeof(Defocus)*ptr->K);
        memset(max_idx ,0,sizeof(single )*ptr->K);
        memset(sum_cc  ,0,sizeof(single )*ptr->K);

        // Apply 3D alignment
        M33f  R_ali;
        Math::eZYZ_Rmat(R_ali,ptr->ptcl.ali_eu[ptr->class_ix]);
        Math::set(Rot,R_ali.transpose());
        ali_data.pre_rotate_reference(Rot,ptr->g_ali,ptr->K,stream);

        // No additional 3D rotation per projection
        M33f  R_eye = Eigen::MatrixXf::Identity(3,3);
        Math::set(Rot,R_eye);
        ali_data.rotate_projections(Rot,ptr->g_ali,ptr->K,stream);
        ali_data.project(vol.ref,bandpass,ptr->K,stream);
        rad_avgr.calculate_FRC(ali_data.prj_c,ptr->K,stream);
        rad_avgr.apply_FRC(ali_data.prj_c,ptr->K,stream);
        ctf_ref.load_buf(ali_data.prj_c,ptr->K,stream);

        float dU,dV,dA;
        float3 delta_w;

        if( astigmatism ) {
            for( dU=-def_range; dU<(def_range+0.5*def_step); dU+=def_step ) {
                delta_w.x  = dU;
                for( dV=-def_range; dV<(def_range+0.5*def_step); dV+=def_step ) {
                    delta_w.y  = dV;
                    for( dA=-ang_range; dA<(ang_range+0.5*ang_step); dA+=ang_step ) {
                        delta_w.z  = dA;
                        ctf_ref.apply_ctf(ali_data.prj_c,delta_w,rad_avgr,ptr,stream);
                        ali_data.apply_bandpass(ptr->ctf_vals,ptr->g_def,bandpass,ptr->K,stream);
                        ali_data.multiply(ss_data.ss_fourier,ptr->K,stream);
                        ali_data.invert_fourier(ptr->K,stream);
                        stream.sync();
                        ali_data.extract_cc(ite_cc,ite_idx,ptr->g_ali,ptr->K,stream);

                        for(int i=0;i<ptr->K;i++) {
                            if( ptr->c_ali.ptr[i].w > 0 ) {
                                sum_cc[i] += ite_cc[i];
                                if( ite_cc[i] > max_cc[i] ) {
                                    def_rslt[i].angle = dA;
                                    def_rslt[i].U = dU;
                                    def_rslt[i].V = dV;
                                    max_cc[i]  = ite_cc[i];
                                    max_idx[i] = ite_idx[i];
                                }
                            }
                        }
                    }
                }
            }
        }
        else {
            for( dU=-def_range; dU<(def_range+0.5*def_step); dU+=def_step ) {
                delta_w.x  = dU;
                delta_w.y  = dU;
                delta_w.z  = 0.0f;
                ctf_ref.apply_ctf(ali_data.prj_c,delta_w,rad_avgr,ptr,stream);
                ali_data.apply_bandpass(ptr->ctf_vals,ptr->g_def,bandpass,ptr->K,stream);
                ali_data.multiply(ss_data.ss_fourier,ptr->K,stream);
                ali_data.invert_fourier(ptr->K,stream);
                stream.sync();
                ali_data.extract_cc(ite_cc,ite_idx,ptr->g_ali,ptr->K,stream);

                for(int i=0;i<ptr->K;i++) {
                    if( ptr->c_ali.ptr[i].w > 0 ) {
                        sum_cc[i] += ite_cc[i];
                        if( ite_cc[i] > max_cc[i] ) {
                            def_rslt[i].angle = 0.0f;
                            def_rslt[i].U = dU;
                            def_rslt[i].V = dU;
                            max_cc[i]  = ite_cc[i];
                            max_idx[i] = ite_idx[i];
                        }
                    }
                }
            }
        }

        single cc_acc=0,wgt_acc=0,cc_cur=0;
        for(int i=0;i<ptr->K;i++) {
            if( ptr->c_ali.ptr[i].w > 0 ) {
                cc_cur  = max_cc[i];
                cc_acc  += cc_cur;
                wgt_acc += ptr->ptcl.prj_w[i];
                update_particle_projection(ptr->ptcl,
                                   def_rslt[i],ali_data.c_pts[max_idx[i]],cc_cur,
                                   i,ptr->ctf_vals.apix);
            }
        }
        ptr->ptcl.ali_cc[ptr->class_ix] = cc_acc/fmax(wgt_acc,1.0);
    }

    void update_particle_projection(Particle&ptcl,const Defocus&delta_def,const Vec3&t,const single cc, const int prj_ix,const float apix) {
        if( ptcl.prj_w[prj_ix] > 0 ) {
            ptcl.prj_cc[prj_ix] = cc;
            ptcl.prj_t[prj_ix].x += t.x*apix;
            ptcl.prj_t[prj_ix].y += t.y*apix;
            ptcl.def[prj_ix].U += delta_def.U;
            ptcl.def[prj_ix].V += delta_def.V;
            ptcl.def[prj_ix].angle += delta_def.angle;
            ptcl.def[prj_ix].score  = cc;

            if( est_dose )
                ptcl.def[prj_ix].ExpFilt = delta_def.ExpFilt;
        }
    }

};

class CtfRefRdrWorker : public Worker {

public:
    ArgsCtfRef::Info *p_info;
    float            *p_stack;
    ParticlesSubset  *p_ptcls;
    Tomogram         *p_tomo;
    RefMap           *p_refs;
    int gpu_ix;
    int max_K;
    int N;
    int M;
    int R;
    int P;
    int pad_type;
    int NP;
    int MP;

    float bp_pad;

    SubstackCrop    ss_cropper;

    CtfRefRdrWorker() {
    }

    ~CtfRefRdrWorker() {
    }

    void setup_global_data(int id,RefMap*in_p_refs,int in_R,int in_max_K,ArgsCtfRef::Info*info,WorkerCommand*in_worker_cmd) {
        worker_id  = id;
        worker_cmd = in_worker_cmd;

        p_info   = info;
        int threads_per_gpu = (info->n_threads) / (info->n_gpu);
        gpu_ix   = info->p_gpu[ id / threads_per_gpu ];
        max_K    = in_max_K;
        pad_type = info->pad_type;

        N = info->box_size;
        M = (N/2)+1;
        P = info->pad_size;

        NP = N + P;
        MP = (NP/2)+1;

        p_refs = in_p_refs;
        R = in_R;
    }

    void setup_working_data(float*stack,ParticlesSubset*ptcls,Tomogram*tomo) {
        p_stack = stack;
        p_ptcls = ptcls;
        p_tomo  = tomo;
        ss_cropper.setup(tomo,N);
        work_progress=0;
    }

protected:
    void main() {

        GPU::set_device(gpu_ix);
        GPU::Stream stream;
        stream.configure();
        work_accumul = 0;
        CtfRefBuffer buffer_a(N,max_K);
        CtfRefBuffer buffer_b(N,max_K);
        PBarrier local_barrier(2);
        DoubleBufferHandler stack_buffer((void*)&buffer_a,(void*)&buffer_b,&local_barrier);

        CtfRefGpuWorker gpu_worker;
        init_processing_worker(gpu_worker,&stack_buffer);

        int current_cmd;

        while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
            switch(current_cmd) {
                case CTF_REF:
                    crop_loop(stack_buffer,stream);
                    break;
                default:
                    break;
            }
        }
        gpu_worker.wait();
    }

    void init_processing_worker(CtfRefGpuWorker&gpu_worker,DoubleBufferHandler*stack_buffer) {
        bp_pad = p_info->fpix_roll/2;
        float bp_scale = ((float)NP)/((float)N);
        gpu_worker.worker_id   = worker_id;
        gpu_worker.worker_cmd  = worker_cmd;
        gpu_worker.gpu_ix      = gpu_ix;
        gpu_worker.p_buffer    = stack_buffer;
        gpu_worker.N           = N;
        gpu_worker.M           = M;
        gpu_worker.P           = P;
        gpu_worker.R           = R;
        gpu_worker.p_refs      = p_refs;
        gpu_worker.use_halves  = p_info->use_halves;
        gpu_worker.pad_type    = pad_type;
        gpu_worker.max_K       = max_K;
        gpu_worker.ssnr.x      = p_info->ssnr_F;
        gpu_worker.ssnr.y      = p_info->ssnr_S;
        gpu_worker.off_par.x   = p_info->off_x;
        gpu_worker.off_par.y   = p_info->off_y;
        gpu_worker.off_par.z   = p_info->off_z;
        gpu_worker.off_par.w   = p_info->off_s;
        gpu_worker.bandpass.x  = fmax(bp_scale*p_info->fpix_min-bp_pad,0.0);
        gpu_worker.bandpass.y  = fmin(bp_scale*p_info->fpix_max+bp_pad,((float)NP)/2);
        gpu_worker.bandpass.z  = sqrt(p_info->fpix_roll);
        gpu_worker.def_range   = p_info->def_range;
        gpu_worker.def_step    = p_info->def_step;
        gpu_worker.ang_range   = p_info->ang_range;
        gpu_worker.ang_step    = p_info->ang_step;
        gpu_worker.est_dose    = p_info->est_dose;
        gpu_worker.astigmatism = p_info->astigmatism;
        gpu_worker.start();
    }

    void crop_loop(DoubleBufferHandler&stack_buffer,GPU::Stream&stream) {
        stack_buffer.WO_sync(EMPTY);
        for(int i=worker_id;i<p_ptcls->n_ptcl;i+=p_info->n_threads) {
            work_progress++;
            work_accumul++;
            CtfRefBuffer*ptr = (CtfRefBuffer*)stack_buffer.WO_get_buffer();
            p_ptcls->get(ptr->ptcl,i);
            read_defocus(ptr);
            crop_substack(ptr,ptr->ptcl.ref_cix());
            if( check_substack(ptr) ) {
                upload(ptr,stream.strm);
                stream.sync();
                stack_buffer.WO_sync(READY);
            }
        }
        stack_buffer.WO_sync(DONE);
    }

    void read_defocus(CtfRefBuffer*ptr) {
        ptr->K = p_tomo->stk_dim.z;

        float lambda = Math::get_lambda( p_tomo->KV );

        ptr->ctf_vals.AC = p_tomo->AC;
        ptr->ctf_vals.CA = sqrt(1-p_tomo->AC*p_tomo->AC);
        ptr->ctf_vals.apix = p_tomo->pix_size;
        ptr->ctf_vals.LambdaPi = M_PI*lambda;
        ptr->ctf_vals.CsLambda3PiH = lambda*lambda*lambda*(p_tomo->CS*1e7)*M_PI/2;

        memcpy( (void**)(ptr->c_def.ptr), (const void**)(ptr->ptcl.def), sizeof(Defocus)*ptr->K  );

        for(int k=0;k<ptr->K;k++) {
            if( ptr->c_def.ptr[k].max_res > 0 ) {
                ptr->c_def.ptr[k].max_res = ((float)NP)*p_tomo->pix_size/ptr->c_def.ptr[k].max_res;
                ptr->c_def.ptr[k].max_res = min(ptr->c_def.ptr[k].max_res+bp_pad,(float)NP/2);
            }
        }
    }

    void crop_substack(CtfRefBuffer*ptr,const int r) {

        V3f pt_tomo,pt_crop;
        M33f R_2D,R_base,R_gpu;

        ptr->r_ix = (p_info->use_halves)?
                        2*r + (ptr->ptcl.half_id()-1): // True
                        r;                             // False

        pt_tomo = get_tomo_position(ptr->ptcl.pos(),ptr->ptcl.ali_t[r]);
        pt_tomo = pt_tomo - p_tomo->tomo_position;

        for(int k=0;k<ptr->K;k++) {
            if( ptr->ptcl.prj_w[k] > 0 ) {

                Math::eZYZ_Rmat(R_2D,ptr->ptcl.prj_eu[k]);
                R_base = R_2D * p_tomo->R[k];

                pt_crop = project_tomo_position(pt_tomo,p_tomo->R[k],p_tomo->t[k],ptr->ptcl.prj_t[k]);
                pt_crop = pt_crop/p_tomo->pix_size + p_tomo->stk_center; /// Angstroms -> pixels

                /// Get subpixel shift and setup data for upload to GPU
                ptr->c_ali.ptr[k].t.x = -(pt_crop(0) - floor(pt_crop(0)));
                ptr->c_ali.ptr[k].t.y = -(pt_crop(1) - floor(pt_crop(1)));
                ptr->c_ali.ptr[k].t.z = 0;
                ptr->c_ali.ptr[k].w = ptr->ptcl.prj_w[k];
                R_gpu = R_base.transpose();
                Math::set( ptr->c_ali.ptr[k].R, R_gpu );

                /// Crop
                if( ss_cropper.check_point(pt_crop) ) {
                    ss_cropper.crop(ptr->c_stk.ptr,p_stack,pt_crop,k);

                    float avg,std;
                    float *ss_ptr = ptr->c_stk.ptr+(k*N*N);
                    Math::get_avg_std(avg,std,ss_ptr,N*N);

                    if( std < SUSAN_FLOAT_TOL || isnan(std) || isinf(std) ) {
                        ptr->c_pad.ptr[k].x = 0;
                        ptr->c_pad.ptr[k].y = 1;
                        ptr->c_ali.ptr[k].w = 0;
                    }
                    else {
                        if( p_info->norm_type == NO_NORM ) {
                            ptr->c_pad.ptr[k].x = avg;
                            ptr->c_pad.ptr[k].y = std;
                        }
                        else if( p_info->norm_type == GAT_RAW ) {
                            Math::anscombe_transform(ss_ptr,N*N);
                            ptr->c_pad.ptr[k].x = 0;
                            ptr->c_pad.ptr[k].y = 1;
                        }
                        else {
                            Math::normalize(ss_ptr,N*N,avg,std);

                            ptr->c_pad.ptr[k].x = 0;
                            ptr->c_pad.ptr[k].y = 1;

                            if( p_info->norm_type == ::ZERO_MEAN ) {
                                ptr->c_pad.ptr[k].y = std;
                            }
                            if( p_info->norm_type == ZERO_MEAN_W_STD ) {
                                ptr->c_pad.ptr[k].y = ptr->ptcl.prj_w[k];
                            }
                            if( p_info->norm_type == GAT_NORMAL ) {
                                Math::generalized_anscombe_transform_zero_mean(ss_ptr,N*N);
                                ptr->c_pad.ptr[k].y = 1;
                            }
                        }
                    }
                }
                else {
                    ptr->c_ali.ptr[k].w = 0;
                }
            }
            else {
                ptr->c_ali.ptr[k].w = 0;
            }
        }
    }

    bool check_substack(CtfRefBuffer*ptr) {
        bool rslt = false;
        for(int k=0;k<ptr->K;k++) {
            if( ptr->c_ali.ptr[k].w > 0 )
                rslt = true;
        }
        return rslt;
    }

    static V3f get_tomo_position(const Vec3&pos_base,const Vec3&shift,bool drift=true) {
        V3f pos_tomo;
        if (drift) {
            pos_tomo(0) = pos_base.x + shift.x;
            pos_tomo(1) = pos_base.y + shift.y;
            pos_tomo(2) = pos_base.z + shift.z;
        }
        else {
            pos_tomo(0) = pos_base.x;
            pos_tomo(1) = pos_base.y;
            pos_tomo(2) = pos_base.z;
        }
        return pos_tomo;
    }

    static V3f project_tomo_position(const V3f &pos_tomo,
                                     const M33f&R_tomo,
                                     const V3f &shift_tomo,
                                     const Vec2&shift_2D,
                                     bool drift=true)
    {
        V3f pos_stack = R_tomo * pos_tomo + shift_tomo;
        if(drift) {
            pos_stack(0) += shift_2D.x;
            pos_stack(1) += shift_2D.y;
        }
        return pos_stack;
    }

    void upload(CtfRefBuffer*ptr,cudaStream_t&strm) {
        GPU::upload_async(ptr->g_stk.ptr,ptr->c_stk.ptr,N*N*max_K,strm);
        GPU::upload_async(ptr->g_pad.ptr,ptr->c_pad.ptr,max_K    ,strm);
        GPU::upload_async(ptr->g_ali.ptr,ptr->c_ali.ptr,max_K    ,strm);
        GPU::upload_async(ptr->g_def.ptr,ptr->c_def.ptr,max_K    ,strm);
    }

};

class CtfRefinerPool : public PoolCoordinator {

public:
    CtfRefRdrWorker  *workers;
    ArgsCtfRef::Info *p_info;
    WorkerCommand    w_cmd;
    RefMap           *p_refs;
    int max_K;
    int N;
    int M;
    int R;
    int P;
    int n_ptcls;
    int NP;
    int MP;

    ProgressReporter progress;

    CtfRefinerPool(ArgsCtfRef::Info*info,References*in_p_refs,int in_max_K,int num_ptcls,StackReader&stkrdr,int in_num_threads)
     : PoolCoordinator(stkrdr,in_num_threads),
       w_cmd(2*in_num_threads+1),
       progress("    Refining particles CTF",num_ptcls)
    {
        workers  = new CtfRefRdrWorker[in_num_threads];
        p_info   = info;
        max_K    = in_max_K;
        n_ptcls  = num_ptcls;
        N = info->box_size;
        M = (N/2)+1;
        P = info->pad_size;
        NP = N+P;
        MP = (NP/2)+1;

        load_references(in_p_refs);
    }

    ~CtfRefinerPool() {
        delete [] p_refs;
        delete [] workers;
    }

protected:
    void load_references(References*in_p_refs) {
        R = in_p_refs->num_refs;
        p_refs = new RefMap[R];
        for(int r=0;r<R;r++) {
            p_refs[r].load(in_p_refs->at(r));
        }
    }

    void coord_init() {
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].setup_global_data(i,p_refs,R,max_K,p_info,&w_cmd);
            workers[i].start();
        }
        progress_start();
    }

    void coord_main(float*stack,ParticlesSubset&ptcls,Tomogram&tomo) {

        w_cmd.presend_sync();
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].setup_working_data(stack,&ptcls,&tomo);
        }
        w_cmd.send_command(CtfRefCmd::CTF_REF);

        show_progress(ptcls.n_ptcl);

        w_cmd.send_command(WorkerCommand::BasicCommands::CMD_IDLE);

    }

    void coord_end() {
        show_done();
        w_cmd.send_command(WorkerCommand::BasicCommands::CMD_END);
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].wait();
        }
    }

    long count_progress() {
        long count = 0;
        for(int i=0;i<p_info->n_threads;i++) {
            count += workers[i].work_progress;
        }
        return count;
    }

    long count_accumul() {
        long count = 0;
        for(int i=0;i<p_info->n_threads;i++) {
            count += workers[i].work_accumul;
        }
        return count;
    }

    virtual void progress_start() {
        progress.start();
    }

    virtual void show_progress(const int ptcls_in_tomo) {
        int cur_progress=0;
        while( (cur_progress=count_progress()) < ptcls_in_tomo ) {
            int total_progress = count_accumul();
            progress.update(total_progress,cur_progress==0);
            sleep(2);
        }
    }

    virtual void show_done() {
        progress.finish();
    }

};

#endif /// CTF_REFINER_H


