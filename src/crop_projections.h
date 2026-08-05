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

#ifndef REC_SUBTOMO_H
#define REC_SUBTOMO_H

#include <iostream>
#include "datatypes.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "pool_coordinator.h"
#include "particles.h"
#include "tomogram.h"
#include "stack_reader.h"
#include "gpu.h"
#include "gpu_kernel.h"
#include "gpu_kernel_ctf.h"
#include "substack_crop.h"
#include "mrc.h"
#include "em.h"
#include "io.h"
#include "points_provider.h"
#include "rec_acc.h"
#include "crop_projections_args.h"

typedef enum {
    CROP_EXEC=1
} CropCmd;

class CropBuffer {

public:
    GPU::GHostSingle c_stk;
    Particle ptcl;
    int      K;

    CropBuffer(int N,int max_k) {
        c_stk.alloc(N*N*max_k);
    }

    ~CropBuffer() {
    }

};

class CropProjectionsWorker : public Worker {

public:
    ArgsCropProjections::Info *p_info;
    float           *p_stack;
    ParticlesSubset *p_ptcls;
    Tomogram        *p_tomo;
    int max_K;
    int N;
    int M;

    SubstackCrop    ss_cropper;

    char stack_file[SUSAN_FILENAME_LENGTH];
    char work_file[SUSAN_FILENAME_LENGTH];
    char data_file[SUSAN_FILENAME_LENGTH];

    CropProjectionsWorker() {
    }

    ~CropProjectionsWorker() {
    }

    void setup_global_data(int id,int in_max_K,ArgsCropProjections::Info*info,WorkerCommand*in_worker_cmd) {
        worker_id  = id;
        worker_cmd = in_worker_cmd;

        p_info   = info;
        max_K    = in_max_K;

        N = info->box_size;
        M = (N/2)+1;
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

        work_accumul = 0;
        CropBuffer buffer(N,max_K);

        int current_cmd;

        sprintf(work_file, "stack_%02d.mrcs"  ,worker_id);
        sprintf(data_file, "%s/stack_%02d.txt",p_info->out_dir,worker_id);
        sprintf(stack_file,"%s/%s"            ,p_info->out_dir,work_file);

        Mrc::SequentialWriter mrc_writer(stack_file,N,1.0);

        FILE*fp_data = fopen(data_file,"w");

        while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
            switch(current_cmd) {
                case CROP_EXEC:
                    crop_loop(buffer,fp_data,mrc_writer);
                    break;
                default:
                    break;
            }
        }

        fclose(fp_data);
    }

    void crop_loop(CropBuffer&buffer,FILE*fp,Mrc::SequentialWriter&mrc_writer) {
        for(int i=worker_id;i<p_ptcls->n_ptcl;i+=p_info->n_threads) {
            p_ptcls->get(buffer.ptcl,i);
            crop_substack(buffer,fp,mrc_writer);
            work_progress++;
            work_accumul++;
        }
    }

    void crop_substack(CropBuffer&ptr,FILE*fp,Mrc::SequentialWriter&mrc_writer) {
        V3f pt_tomo,pt_crop,eu_ZYZ;
        M33f R_2D,R_3D,R_base,R_gpu;

        float avg,std;
        float *ss_ptr;

        int r = ptr.ptcl.ref_cix();

        calc_R(R_3D,ptr.ptcl.ali_eu[r],true);

        pt_tomo = get_tomo_position(ptr.ptcl.pos(),ptr.ptcl.ali_t[r]);
        pt_tomo = pt_tomo - p_tomo->tomo_position;

        for(int k=0;k<p_tomo->stk_dim.z;k++) {
            if( ptr.ptcl.prj_w[k] <= 0 )
                continue;

            calc_R(R_2D,ptr.ptcl.prj_eu[k],true);
            R_base = R_2D * p_tomo->R[k];

            pt_crop = project_tomo_position(pt_tomo,p_tomo->R[k],p_tomo->t[k],ptr.ptcl.prj_t[k]);
            pt_crop = pt_crop/p_tomo->pix_size + p_tomo->stk_center; /// Angstroms -> pixels

            if( !ss_cropper.check_point(pt_crop) )
                continue;

            ss_cropper.crop(ptr.c_stk.ptr,p_stack,pt_crop,k);
            ss_ptr = ptr.c_stk.ptr+(k*N*N);
            Math::get_avg_std(avg,std,ss_ptr,N*N);

            if( std < SUSAN_FLOAT_TOL || isnan(std) || isinf(std) )
                continue;

            if( p_info->norm_type == NO_NORM )         std = -1.0;
            if( p_info->norm_type == ZERO_MEAN )       std =  1.0;

            if( std > 0 )        Math::normalize(ss_ptr,N*N,avg,std);
            if( p_info->invert ) Math::mul(ss_ptr,-1.0f,N*N);

            /// SAVE
            mrc_writer.apix = p_tomo->pix_size;
            mrc_writer.push_frame(ss_ptr);

            R_gpu = R_base*R_3D;
            Math::Rmat_eZYZ(eu_ZYZ,R_gpu);
            /// rlnImageName
            fprintf(fp,"%08d@%s  ",mrc_writer.K,work_file);

            /// rlnCoordinate{X,Y,Z}
            fprintf(fp,"%4d  %4d  %4d  ",pt_crop(0),pt_crop(1),k);

            /// rlnMicrographName
            fprintf(fp,"%s  ",p_tomo->stk_name);

            /// rlnDefocus{U,V,Angle} rlnPhaseShift
            fprintf(fp,"%5.0f  %5.0f  %5.1f  %5.1f  ",ptr.ptcl.def[k].U,ptr.ptcl.def[k].V,ptr.ptcl.def[k].angle,RAD2DEG*ptr.ptcl.def[k].ph_shft);

            /// rlnAngle{Rot,Tilt,Psi}
            fprintf(fp,"%7.2f  %7.2f  %7.2f  ",-RAD2DEG*eu_ZYZ(2),-RAD2DEG*eu_ZYZ(1),-RAD2DEG*eu_ZYZ(0));

            /// rlnOrigin{X,Y} rlnRandomSubset
            fprintf(fp,"0 0 %d\n",ptr.ptcl.half_id());

        }
    }

    void calc_R(M33f&R_out,const Vec3&eu_in,bool use_ali) {
        V3f eu_ZYZ;

        if( use_ali ) {
            eu_ZYZ(0) = eu_in.x;
            eu_ZYZ(1) = eu_in.y;
            eu_ZYZ(2) = eu_in.z;
        }
        else {
            eu_ZYZ(0) = 0;
            eu_ZYZ(1) = 0;
            eu_ZYZ(2) = 0;
        }
        Math::eZYZ_Rmat(R_out,eu_ZYZ);
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
};

class CropProjectionsPool : public PoolCoordinator {

public:
    CropProjectionsWorker     *workers;
    ArgsCropProjections::Info *p_info;
    WorkerCommand w_cmd;
    int max_K;
    int N;
    int M;
    int n_ptcls;

    Math::Timing timer;

    char progress_buffer[68];
    char progress_clear [69];

    CropProjectionsPool(ArgsCropProjections::Info*info,int in_max_K,int num_ptcls,StackReader&stkrdr,int in_num_threads)
     : PoolCoordinator(stkrdr,in_num_threads), w_cmd(in_num_threads+1) {
        workers  = new CropProjectionsWorker[in_num_threads];
        p_info   = info;
        max_K    = in_max_K;
        n_ptcls  = num_ptcls;
        N = info->box_size;
        M = (N/2)+1;
        memset(progress_buffer,' ',66);
        memset(progress_clear,'\b',66);
        progress_buffer[66] = 0;
        progress_buffer[67] = 0;
        progress_clear [66] = '\r';
        progress_clear [67] = 0;

        IO::create_dir(info->out_dir);
    }

    ~CropProjectionsPool() {
        delete [] workers;
    }

protected:

    void coord_init() {
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].setup_global_data(i,max_K,p_info,&w_cmd);
            workers[i].start();
        }
        progress_start();
    }

    void coord_main(float*stack,ParticlesSubset&ptcls,Tomogram&tomo) {

        w_cmd.presend_sync();
        for(int i=0;i<p_info->n_threads;i++) {
            workers[i].setup_working_data(stack,&ptcls,&tomo);
        }
        w_cmd.send_command(CropCmd::CROP_EXEC);

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

    void progress_start() {
        timer.tic();
        sprintf(progress_buffer,"        Cropping projections: Buffering...");
        int n = strlen(progress_buffer);
        progress_buffer[n]  = ' ';
        progress_buffer[65] = 0;
        printf("%s", progress_buffer);
        fflush(stdout);
    }

    void show_progress(const int ptcls_in_tomo) {
        int cur_progress=0;
        while( (cur_progress=count_progress()) < ptcls_in_tomo ) {
            memset(progress_buffer,' ',66);
            if( cur_progress > 0 ) {
                int progress = count_accumul();
                float progress_percent = 100*(float)progress/float(n_ptcls);
                sprintf(progress_buffer,"        Cropping projections: %6.2f%%%%",progress_percent);
                int n = strlen(progress_buffer);
                add_etc(progress_buffer+n,progress,n_ptcls);
            }
            else {
                sprintf(progress_buffer,"        Cropping projections: Buffering...");
                int n = strlen(progress_buffer);
                progress_buffer[n]  = ' ';
                progress_buffer[65] = 0;
            }
            printf("%s", progress_clear);
            fflush(stdout);
            printf("%s", progress_buffer);
            fflush(stdout);
            sleep(1);
        }
    }

    void show_done() {
        memset(progress_buffer,' ',66);
        sprintf(progress_buffer,"        Cropping projections: 100.00%%%%");
        int n = strlen(progress_buffer);
        progress_buffer[n] = ' ';
        printf("%s", progress_clear);
        printf("%s", progress_buffer);
        printf("\n");
        fflush(stdout);
    }

    void add_etc(char*buffer,int progress,int total) {

        int days,hours,mins,secs;

        timer.get_etc(days,hours,mins,secs,progress,total);

        if( days > 0 ) {
            sprintf(buffer," (ETC: %dd %02d:%02d:%02d)",days,hours,mins,secs);
            int n = strlen(buffer);
            buffer[n] = ' ';
        }
        else {
            sprintf(buffer," (ETC: %02d:%02d:%02d)",hours,mins,secs);
            int n = strlen(buffer);
            buffer[n] = ' ';
        }
    }

};

#endif /// REC_SUBTOMO_H


