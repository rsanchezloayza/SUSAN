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

#ifndef RECONSTRUCT_H
#define RECONSTRUCT_H

#include <iostream>
#include <vector>
#include "datatypes.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "pool_coordinator.h"
#include "particles.h"
#include "tomogram.h"
#include "stack_reader.h"
#include "gpu.h"
#include "substack_crop.h"
#include "mrc.h"
#include "rec_acc.h"
#include "reconstruct_args.h"
#include "progress.h"

typedef enum {
    REC_EXEC=1
} RecCmd;

class RecBuffer {

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
    int      h_ix;

    RecBuffer(int N,int max_k) {
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

    ~RecBuffer() {
    }

};

class RecGpuWorker : public Worker {

public:
    int gpu_ix;
    int N;
    int M;
    int P;
    int R;
    int pad_type;
    int ctf_type;
    int grid_type;
    int max_K;
    float3  bandpass;
    float2  ssnr; /// x=F; y=S;
    double2 **c_acc;
    double  **c_wgt;
    DoubleBufferHandler *p_buffer;

    RecGpuWorker() {
    }

    ~RecGpuWorker() {
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
        RecSubstack ss_data(M,N,max_K,P,stream);
        std::vector<RecAcc> vols_vec(R);
        RecAcc* vols = vols_vec.data();
        for(int r=0;r<R;r++)
            vols[r].alloc(MP,NP,max_K);

        while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
            switch(current_cmd) {
                case REC_EXEC:
                    insert_loop(vols,ss_data,stream);
                    break;
                default:
                    break;
            }
        }

        for(int r=0;r<R;r++)
            download_vol(c_acc[r],c_wgt[r],vols[r]);

        GPU::sync();
    }

    void insert_loop(RecAcc*vols,RecSubstack&ss_data,GPU::Stream&stream) {
        p_buffer->RO_sync();
        while( p_buffer->RO_get_status() > DONE ) {
            if( p_buffer->RO_get_status() == READY ) {
                RecBuffer*ptr = (RecBuffer*)p_buffer->RO_get_buffer();
                add_data(ss_data,ptr,stream);
                correct_ctf(ss_data,ptr,stream);
                insert_vol(vols[ptr->r_ix],ss_data,ptr,stream);
                stream.sync();
            }
            p_buffer->RO_sync();
        }
    }

    void add_data(RecSubstack&ss_data,RecBuffer*ptr,GPU::Stream&stream) {
        if( pad_type == PAD_ZERO )
            ss_data.pad_zero(stream);
        if( pad_type == PAD_GAUSSIAN )
            ss_data.pad_normal(ptr->g_pad,ptr->K,stream);

        ss_data.add_data(ptr->g_stk,ptr->g_ali,ptr->K,stream);
    }

    void correct_ctf(RecSubstack&ss_data,RecBuffer*ptr,GPU::Stream&stream) {
        if( ctf_type == INV_NO_INV )
            ss_data.set_no_ctf(ptr->g_def,bandpass,ptr->K,stream);
        if( ctf_type == INV_PHASE_FLIP )
            ss_data.set_phase_flip(ptr->ctf_vals,ptr->g_def,bandpass,ptr->K,stream);
        if( ctf_type == INV_WIENER )
            ss_data.set_wiener(ptr->ctf_vals,ptr->g_def,bandpass,ptr->K,stream);
        if( ctf_type == INV_WIENER_SSNR )
            ss_data.set_wiener_ssnr(ptr->ctf_vals,ptr->g_def,bandpass,ssnr,ptr->K,stream);
        if( ctf_type == INV_WIENER_ARCTAN )
            ss_data.set_wiener(ptr->ctf_vals,ptr->g_def,bandpass,ptr->K,stream);
        if( ctf_type == INV_WIENER_LOGISTIC )
            ss_data.set_wiener(ptr->ctf_vals,ptr->g_def,bandpass,ptr->K,stream);
    }

    void insert_vol(RecAcc&vol,RecSubstack&ss_data,RecBuffer*ptr,GPU::Stream&stream) {
        if(grid_type == GRIDDING_LINEAR_FWD)
            vol.insert_linear_fwd(ss_data.ss_tex,ss_data.ss_ctf,ptr->g_ali,bandpass,ptr->K,stream);
        else if(grid_type == GRIDDING_KAISER_BESSEL_FWD)
            vol.insert_kaiser_bessel_fwd(ss_data.ss_tex,ss_data.ss_ctf,ptr->g_ali,bandpass,ptr->K,stream);
    }

    void download_vol(double2*p_acc,double*p_wgt,RecAcc&vol) {
        cudaMemcpy( (void*)(p_acc), (const void*)vol.vol_acc.ptr, sizeof(double2)*NP*NP*MP, cudaMemcpyDeviceToHost);
        cudaMemcpy( (void*)(p_wgt), (const void*)vol.vol_wgt.ptr, sizeof(double )*NP*NP*MP, cudaMemcpyDeviceToHost);
    }
};

class RecRdrWorker : public Worker {

public:
    ArgsRec::Info   *p_info;
    float           *p_stack;
    ParticlesSubset *p_ptcls;
    Tomogram        *p_tomo;
    int gpu_ix;
    int max_K;
    int N;
    int M;
    int R;
    int P;
    int pad_type;
    int NP;
    int MP;

    double2 **c_acc;
    double  **c_wgt;

    float bp_pad;

    SubstackCrop    ss_cropper;

    RecRdrWorker() {
        c_acc = NULL;
        c_wgt = NULL;
    }

    ~RecRdrWorker() {
        if( c_acc != NULL ) {
            for(int r=0;r<R;r++) {
                delete [] c_acc[r];
            }
            delete [] c_acc;
        }

        if( c_wgt != NULL ) {
            for(int r=0;r<R;r++) {
                delete [] c_wgt[r];
            }
            delete [] c_wgt;
        }
    }

    void setup_global_data(int id,int n_refs,int in_max_K,ArgsRec::Info*info,WorkerCommand*in_worker_cmd) {
        worker_id  = id;
        worker_cmd = in_worker_cmd;

        p_info   = info;
        int threads_per_gpu = (info->n_threads) / (info->n_gpu);
        gpu_ix   = info->p_gpu[ id / threads_per_gpu ];
        max_K    = in_max_K;
        pad_type = info->pad_type;


        N = info->box_size;
        M = (N/2)+1;
        R = n_refs;
        P = info->pad_size;

        NP = N + P;
        MP = (NP/2)+1;

        c_acc = new double2*[R];
        c_wgt = new double *[R];
        for(int r=0;r<R;r++) {
            c_acc[r] = new double2[NP*MP*NP];
            c_wgt[r] = new double [NP*MP*NP];
        }
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
        RecBuffer buffer_a(N,max_K);
        RecBuffer buffer_b(N,max_K);
        PBarrier local_barrier(2);
        DoubleBufferHandler stack_buffer((void*)&buffer_a,(void*)&buffer_b,&local_barrier);

        RecGpuWorker gpu_worker;
        init_processing_worker(gpu_worker,&stack_buffer);

        int current_cmd;

        while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
            switch(current_cmd) {
                case REC_EXEC:
                    if(p_info->ignore_ref)
                        crop_loop_ignore_ref(stack_buffer,stream);
                    else
                        crop_loop(stack_buffer,stream);
                    break;
                default:
                    break;
            }
        }
        gpu_worker.wait();
    }

    void init_processing_worker(RecGpuWorker&gpu_worker,DoubleBufferHandler*stack_buffer) {
        float bp_scale = ((float)NP)/((float)N);
        bp_pad = bp_scale*p_info->fpix_roll/2;
        gpu_worker.worker_id  = worker_id;
        gpu_worker.worker_cmd = worker_cmd;
        gpu_worker.gpu_ix     = gpu_ix;
        gpu_worker.p_buffer   = stack_buffer;
        gpu_worker.N          = N;
        gpu_worker.M          = M;
        gpu_worker.P          = P;
        gpu_worker.R          = R;
        gpu_worker.pad_type   = pad_type;
        gpu_worker.ctf_type   = p_info->ctf_type;
        gpu_worker.grid_type  = p_info->grid_type;
        gpu_worker.max_K      = max_K;
        gpu_worker.c_acc      = c_acc;
        gpu_worker.c_wgt      = c_wgt;
        gpu_worker.bandpass.x = fmax(bp_scale*p_info->fpix_min-bp_pad,0.0);
        gpu_worker.bandpass.y = fmin(bp_scale*p_info->fpix_max+bp_pad,((float)NP)/2);
        gpu_worker.bandpass.z = bp_scale*sqrt(p_info->fpix_roll);
        gpu_worker.ssnr.x     = p_info->ssnr_F;
        gpu_worker.ssnr.y     = p_info->ssnr_S;
        gpu_worker.start();
    }

    void crop_loop(DoubleBufferHandler&stack_buffer,GPU::Stream&stream) {
        stack_buffer.WO_sync(EMPTY);
        for(int i=worker_id;i<p_ptcls->n_ptcl;i+=p_info->n_threads) {
            RecBuffer*ptr = (RecBuffer*)stack_buffer.WO_get_buffer();
            p_ptcls->get(ptr->ptcl,i);
            if( check_reference(ptr) ) {
                read_defocus(ptr);
                crop_substack(ptr);
                if( check_substack(ptr) ) {
                    upload(ptr,stream.strm);
                    stream.sync();
                    stack_buffer.WO_sync(READY);
                }
            }
            work_progress++;
            work_accumul++;
        }
        stack_buffer.WO_sync(DONE);
    }

    void crop_loop_ignore_ref(DoubleBufferHandler&stack_buffer,GPU::Stream&stream) {
        stack_buffer.WO_sync(EMPTY);
        for(int i=worker_id;i<p_ptcls->n_ptcl;i+=p_info->n_threads) {
            for(int r=0;r<R;r++) {
                RecBuffer*ptr = (RecBuffer*)stack_buffer.WO_get_buffer();
                p_ptcls->get(ptr->ptcl,i);
                if( check_reference(ptr,r) ) {
                    read_defocus(ptr);
                    crop_substack(ptr,r);
                    if( check_substack(ptr) ) {
                        upload(ptr,stream.strm);
                        stream.sync();
                        stack_buffer.WO_sync(READY);
                    }
                }
            }
            work_progress++;
            work_accumul++;
        }
        stack_buffer.WO_sync(DONE);
    }

    void read_defocus(RecBuffer*ptr) {
        ptr->K = p_tomo->stk_dim.z;

        float lambda = Math::get_lambda( p_tomo->KV );

        ptr->ctf_vals.AC = p_tomo->AC;
        ptr->ctf_vals.CA = sqrt(1-p_tomo->AC*p_tomo->AC);
        ptr->ctf_vals.apix = p_tomo->pix_size;
        ptr->ctf_vals.LambdaPi = M_PI*lambda;
        ptr->ctf_vals.CsLambda3PiH = lambda*lambda*lambda*(p_tomo->CS*1e7)*M_PI/2;

        memcpy( (void*)(ptr->c_def.ptr), (const void*)(ptr->ptcl.def), sizeof(Defocus)*ptr->K  );

        for(int k=0;k<ptr->K;k++) {
            if( ptr->c_def.ptr[k].max_res > 0 ) {
                ptr->c_def.ptr[k].max_res = ((float)NP)*p_tomo->pix_size/ptr->c_def.ptr[k].max_res;
                ptr->c_def.ptr[k].max_res = min(ptr->c_def.ptr[k].max_res+bp_pad,(float)NP/2);
            }
        }
    }

    bool check_reference(RecBuffer*ptr,int r=-1) {
        if(r<0)
            r = ptr->ptcl.ref_cix();

        if( p_info->rec_halves )
            ptr->r_ix = 2*r + (ptr->ptcl.half_id()-1);
        else
            ptr->r_ix = r;

        if( p_info->rec_halves )
            return ( ptr->ptcl.ali_w[r] != 0 ) && ( ptr->ptcl.half_id() > 0 );
        else
            return ( ptr->ptcl.ali_w[r] ) != 0;
    }

    void crop_substack(RecBuffer*ptr,int r=-1) {
        V3f pt_tomo,pt_crop;
        M33f R_2D,R_3D,R_base,R_gpu;

        if ( r<0 ) r = ptr->ptcl.ref_cix();

        pt_tomo = get_tomo_position(ptr->ptcl.pos(),ptr->ptcl.ali_t[r]);
        pt_tomo = pt_tomo - p_tomo->tomo_position;

        Math::eZYZ_Rmat(R_3D,ptr->ptcl.ali_eu[r]);

        for(int k=0;k<ptr->K;k++) {
            ptr->c_ali.ptr[k].w = 0;

            if( ptr->ptcl.prj_w[k] > 0 ) {

                Math::eZYZ_Rmat(R_2D,ptr->ptcl.prj_eu[k]);
                R_base = R_2D * p_tomo->R[k];

                pt_crop = project_tomo_position(pt_tomo,p_tomo->R[k],p_tomo->t[k],ptr->ptcl.prj_t[k]);
                if( p_tomo->pix_size == 0 ) { ptr->c_ali.ptr[k].w = 0; continue; }
                pt_crop = pt_crop/p_tomo->pix_size + p_tomo->stk_center; /// Angstroms -> pixels

                /// Get subpixel shift and setup data for upload to GPU
                ptr->c_ali.ptr[k].t.x = -(pt_crop(0) - floor(pt_crop(0)));
                ptr->c_ali.ptr[k].t.y = -(pt_crop(1) - floor(pt_crop(1)));
                ptr->c_ali.ptr[k].t.z = 0;
                ptr->c_ali.ptr[k].w = ptr->ptcl.prj_w[k];
                R_gpu = (R_base*R_3D).transpose();
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
                        else {
                            if( p_info->norm_type == ::ZERO_MEAN ) {
                                Math::normalize(ss_ptr,N*N,avg,1.0);
                                ptr->c_pad.ptr[k].x = 0;
                                ptr->c_pad.ptr[k].y = std;
                            }

                            if( p_info->norm_type == ::ZERO_MEAN_1_STD ) {
                                Math::normalize(ss_ptr,N*N,avg,std);
                                ptr->c_pad.ptr[k].x = 0;
                                ptr->c_pad.ptr[k].y = 1.0;
                            }

                            if( p_info->norm_type == ZERO_MEAN_W_STD ) {
                                Math::normalize(ss_ptr,N*N,avg,std/ptr->ptcl.prj_w[k]);
                                ptr->c_pad.ptr[k].x = 0;
                                ptr->c_pad.ptr[k].y = ptr->ptcl.prj_w[k];
                            }

                            if( p_info->norm_type == GAT_NORMAL ) {
                                Math::generalized_anscombe_transform_zero_mean(ss_ptr,N*N);
                                ptr->c_pad.ptr[k].x = 0;
                                ptr->c_pad.ptr[k].y = 1;
                            }
                        }

                        /// Set projection weight
                        if( p_info->wgt_type == WGT_NONE )
                            ptr->c_ali.ptr[k].w = (ptr->ptcl.prj_w[k] > 0);
                        if( p_info->wgt_type == WGT_3D )
                            ptr->c_ali.ptr[k].w = ptr->ptcl.ali_w[r];
                        if( p_info->wgt_type == WGT_2D )
                            ptr->c_ali.ptr[k].w = ptr->ptcl.prj_w[k];
                        if( p_info->wgt_type == WGT_3DCC )
                            ptr->c_ali.ptr[k].w = ptr->ptcl.ali_cc[r];
                        if( p_info->wgt_type == WGT_2DCC )
                            ptr->c_ali.ptr[k].w = ptr->ptcl.prj_cc[k];


                    } /// if( std < SUSAN_FLOAT_TOL || isnan(std) || isinf(std) )

                }
                else {
                    ptr->c_ali.ptr[k].w = 0;
                } /// if( ss_cropper.check_point(pt_crop) )
            }
            else {
                ptr->c_ali.ptr[k].w = 0;
            } /// if( ptr->ptcl.prj_w[k] > 0 )

        } /// for(int k=0;k<ptr->K;k++)
    }

    bool check_substack(RecBuffer*ptr) {
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

    void upload(RecBuffer*ptr,cudaStream_t&strm) {
        GPU::upload_async(ptr->g_stk.ptr,ptr->c_stk.ptr,N*N*max_K,strm);
        GPU::upload_async(ptr->g_pad.ptr,ptr->c_pad.ptr,max_K    ,strm);
        GPU::upload_async(ptr->g_ali.ptr,ptr->c_ali.ptr,max_K    ,strm);
        GPU::upload_async(ptr->g_def.ptr,ptr->c_def.ptr,max_K    ,strm);
    }

};

class RecPool : public PoolCoordinator {

public:
    RecRdrWorker  *workers;
    ArgsRec::Info *p_info;
    WorkerCommand w_cmd;
    int max_K;
    int N;
    int M;
    int R;
    int P;
    int n_ptcls;
    int NP;
    int MP;

    ProgressReporter progress;

    RecPool(ArgsRec::Info*info,int n_refs,int in_max_K,int num_ptcls,StackReader&stkrdr,int in_num_threads)
     : PoolCoordinator(stkrdr,in_num_threads),
        w_cmd(2*in_num_threads+1),
        progress("    Filling fourier space",num_ptcls)
    {
        workers  = new RecRdrWorker[in_num_threads];
        p_info   = info;
        max_K    = in_max_K;
        n_ptcls  = num_ptcls;
        N = info->box_size;
        M = (N/2)+1;
        R = n_refs;
        P = info->pad_size;
        NP = N+P;
        MP = (NP/2)+1;
        if( info->rec_halves )
            R = 2*R;

    }

    ~RecPool() {
        delete [] workers;
    }

protected:

    void coord_init() {
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].setup_global_data(i,R,max_K,p_info,&w_cmd);
            workers[i].start();
        }
        progress_start();
    }

    void coord_main(float*stack,ParticlesSubset&ptcls,Tomogram&tomo) {

        w_cmd.presend_sync();
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].setup_working_data(stack,&ptcls,&tomo);
        }
        w_cmd.send_command(RecCmd::REC_EXEC);

        show_progress(ptcls.n_ptcl);

        w_cmd.send_command(WorkerCommand::BasicCommands::CMD_IDLE);

    }

    void coord_end() {
        show_done();
        w_cmd.send_command(WorkerCommand::BasicCommands::CMD_END);
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].wait();
        }

        /// The reader has signalled DONE before we reach coord_end, so the
        /// full-stack buffers are no longer needed. Release them here to free
        /// host RAM before the memory-heavy gather/reconstruct stage.
        stkrdr.free_buffers();

        gather_results();
        reconstruct_results();
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

    void gather_results() {
        int l = NP*NP*MP;
        for(int i=1;i<p_info->n_threads;i++) {
            for(int r=0;r<R;r++) {
                Math::sum(workers[0].c_acc[r],workers[i].c_acc[r],l);
                Math::sum(workers[0].c_wgt[r],workers[i].c_wgt[r],l);
            }
        }
    }

    virtual void reconstruct_results() {
        GPU::set_device( p_info->p_gpu[0] );

        GPU::GArrDouble  p_wgt;
        GPU::GArrDouble2 p_acc;
        GPU::GArrSingle  p_vol;
        float*vol;

        p_wgt.alloc((size_t)MP*NP*NP);
        p_acc.alloc((size_t)MP*NP*NP);
        p_vol.alloc((size_t)N*N*N);
        vol = new float[(size_t)N*N*N];

        if( p_info->rec_halves )
            reconstruct_halves(vol,p_vol,p_acc,p_wgt);
        else
            reconstruct_maps(vol,p_vol,p_acc,p_wgt);

        delete [] vol;
    }

    void reconstruct_maps(float*vol,GPU::GArrSingle&p_vol,GPU::GArrDouble2&p_acc,GPU::GArrDouble&p_wgt) {
        char out_file[SUSAN_FILENAME_LENGTH];
        for(int r=0;r<R;r++) {
            sprintf(out_file,"%s_class%03d.mrc",p_info->out_pfx,r+1);
            if( p_info->verbosity != VERBOSITY_MINIMAL ) {
                printf("        Reconstructing %s ... ",out_file);
                fflush(stdout);
            }
            else {
                printf("    Class %3d:",r+1);
                fflush(stdout);
            }
            reconstruct_upload(workers[0].c_acc[r],workers[0].c_wgt[r],p_acc,p_wgt);
            reconstruct_sym(p_acc,p_wgt);
            reconstruct_invert(p_wgt);
            reconstruct_core(p_vol,p_acc,p_wgt);
            reconstruct_download(vol,p_vol);
            Mrc::write(vol,N,N,N,out_file);
            Mrc::set_apix(out_file,stkrdr.tomos->at(0).pix_size,N,N,N);
            Mrc::set_as_volume(out_file);
            if( p_info->verbosity != VERBOSITY_MINIMAL ) {
                printf(" Done.\n"); fflush(stdout);
            }
            else {
                printf(" full map.\n"); fflush(stdout);
            }
        }
    }

    void reconstruct_halves(float*vol,GPU::GArrSingle&p_vol,GPU::GArrDouble2&p_acc,GPU::GArrDouble&p_wgt) {
        int l = NP*NP*MP;
        char out_file[SUSAN_FILENAME_LENGTH];
        for(int r=0;r<R/2;r++) {

            sprintf(out_file,"%s_class%03d_half1.mrc",p_info->out_pfx,r+1);
            if( p_info->verbosity != VERBOSITY_MINIMAL ) {
                printf("        Reconstructing %s ... ",out_file); fflush(stdout);
            }
            else {
                printf("    Class %3d:",r+1);
                fflush(stdout);
            }
            reconstruct_upload(workers[0].c_acc[2*r  ],workers[0].c_wgt[2*r  ],p_acc,p_wgt);
            reconstruct_sym(p_acc,p_wgt);
            reconstruct_invert(p_wgt);
            reconstruct_core(p_vol,p_acc,p_wgt);
            reconstruct_download(vol,p_vol);
            Mrc::write(vol,N,N,N,out_file);
            Mrc::set_apix(out_file,stkrdr.tomos->at(0).pix_size,N,N,N);
            Mrc::set_as_volume(out_file);
            if( p_info->verbosity != VERBOSITY_MINIMAL ) {
                printf(" Done.\n"); fflush(stdout);
            }
            else {
                printf(" halfmap 1"); fflush(stdout);
            }

            sprintf(out_file,"%s_class%03d_half2.mrc",p_info->out_pfx,r+1);
            if( p_info->verbosity != VERBOSITY_MINIMAL ) {
                printf("        Reconstructing %s ... ",out_file); fflush(stdout);
            }
            reconstruct_upload(workers[0].c_acc[2*r+1],workers[0].c_wgt[2*r+1],p_acc,p_wgt);
            reconstruct_sym(p_acc,p_wgt);
            reconstruct_invert(p_wgt);
            reconstruct_core(p_vol,p_acc,p_wgt);
            reconstruct_download(vol,p_vol);
            Mrc::write(vol,N,N,N,out_file);
            Mrc::set_apix(out_file,stkrdr.tomos->at(0).pix_size,N,N,N);
            Mrc::set_as_volume(out_file);
            if( p_info->verbosity != VERBOSITY_MINIMAL ) {
                printf(" Done.\n"); fflush(stdout);
            }
            else {
                printf(", halfmap 2"); fflush(stdout);
            }

            Math::sum(workers[0].c_acc[2*r],workers[0].c_acc[2*r+1],l);
            Math::sum(workers[0].c_wgt[2*r],workers[0].c_wgt[2*r+1],l);
            sprintf(out_file,"%s_class%03d.mrc",p_info->out_pfx,r+1);
            if( p_info->verbosity != VERBOSITY_MINIMAL ) {
                printf("        Reconstructing %s ... ",out_file); fflush(stdout);
            }
            reconstruct_upload(workers[0].c_acc[2*r  ],workers[0].c_wgt[2*r  ],p_acc,p_wgt);
            reconstruct_sym(p_acc,p_wgt);
            reconstruct_invert(p_wgt);
            reconstruct_core(p_vol,p_acc,p_wgt);
            reconstruct_download(vol,p_vol);
            Mrc::write(vol,N,N,N,out_file);
            Mrc::set_apix(out_file,stkrdr.tomos->at(0).pix_size,N,N,N);
            Mrc::set_as_volume(out_file);
            if( p_info->verbosity != VERBOSITY_MINIMAL ) {
                printf(" Done.\n"); fflush(stdout);
            }
            else {
                printf(", full map.\n"); fflush(stdout);
            }
        }
    }

    void reconstruct_upload(double2*c_acc,double*c_wgt,GPU::GArrDouble2&p_acc,GPU::GArrDouble&p_wgt) {
        cudaMemcpy((void*)p_acc.ptr,(const void*)c_acc,sizeof(double2)*MP*NP*NP,cudaMemcpyHostToDevice);
        cudaMemcpy((void*)p_wgt.ptr,(const void*)c_wgt,sizeof(double )*MP*NP*NP,cudaMemcpyHostToDevice);
    }

    void reconstruct_sym(GPU::GArrDouble2&p_acc,GPU::GArrDouble&p_wgt) {
        RecSym sym(MP,NP,p_info->sym);
        sym.apply_sym(p_acc,p_wgt);
    }

    void reconstruct_invert(GPU::GArrDouble&p_wgt) {
        float frq_scale = float(NP)/float(N);
        RecInvWgt inv_wgt(NP,MP,p_info->w_inv_ite,p_info->w_inv_std);
        inv_wgt.invert(p_wgt,p_info->ctf_type,p_info->ssnr_S*frq_scale,p_info->ssnr_F*frq_scale);
    }

    void reconstruct_core(GPU::GArrSingle&p_vol,GPU::GArrDouble2&p_acc,GPU::GArrDouble&p_wgt) {
        RecInvVol inv_vol(N,P);
        inv_vol.apply_inv_wgt(p_acc,p_wgt);
        if( p_info->boost_low_fq_scale > 0 ) {
            float factor = NP;
            factor = factor/N;
            inv_vol.boost_low_freq(p_info->boost_low_fq_scale,
                                   p_info->boost_low_fq_value*factor,
                                   p_info->boost_low_fq_decay*factor);
        }
        inv_vol.invert_and_extract(p_vol,p_info->grid_type);
    }

    void reconstruct_download(float*vol,GPU::GArrSingle&p_vol) {
        cudaMemcpy((void*)vol,(const void*)p_vol.ptr,sizeof(float)*N*N*N,cudaMemcpyDeviceToHost);
        if( p_info->norm_output ) {
            float avg,std;
            Math::get_avg_std(avg,std,vol,N*N*N);
            if( !Math::normalize(vol,N*N*N,avg,std,1.0) ) {
                Math::randn(vol,N*N*N);
                if( p_info->verbosity != VERBOSITY_MINIMAL ) {
                    printf("(Empty, filling with noise)");
                    fflush(stdout);
                }
            }
        }
    }

};

#endif /// RECONSTRUCT_H


