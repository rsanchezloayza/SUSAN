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

#ifndef ALIGNER_H
#define ALIGNER_H

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
#include "substack_crop.h"
#include "angles_provider.h"
#include "ref_ali.h"
#include "aligner_args.h"
#include "progress.h"

#include "Eigen/Geometry"
#include <Eigen/Eigenvalues>
using namespace Eigen;

typedef enum {
    ALI_3D=1,
    ALI_2D
} AliCmd;

typedef enum {
    TM_NONE=0,
    TM_PYTHON,
    TM_MATLAB,
    TM_CSV
} TEMPLATE_MATCHING_OUTPUT;

class AliBuffer {

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
    int K;
    int r_ix;
    int class_ix;
    int tomo_pos_x;
    int tomo_pos_y;
    int tomo_pos_z;

    float crowther_limit;

    AliBuffer(int N,int max_k) {
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

    ~AliBuffer() {
    }

    void set_tomo_pos(const V3f&pos_tomo,const V3f&tomo_center,const float pix_size) {
        V3f tmp = (pos_tomo/pix_size) + tomo_center;
        tomo_pos_x = (int)roundf( tmp(0) );
        tomo_pos_y = (int)roundf( tmp(1) );
        tomo_pos_z = (int)roundf( tmp(2) );
    }
};

class TemplateMatchingReporter {

    int  tm_type;
    int  tm_dim;
    FILE *fp;

    float *c_cc;
    int   num_points;
    int   max_K;
    int   n_cc;
    float sigma;

    float *p_avg;
    float *p_std;
    float *p_cnt;

    int *p_x;
    int *p_y;
    int *p_z;

    int block_id;

public:
    TemplateMatchingReporter(const Vec3*c_pts,int n_pts, int K, int dim, const float in_sigma=0) {
        tm_type    = TM_NONE;
        tm_dim     = dim;
        num_points = n_pts;
        max_K      = K;

        if(dim==2){       //2D alignment
            n_cc = n_pts * K;
        }
        else if (dim==3){ //3D alignment
            n_cc = n_pts;
        }

        c_cc  = new float[n_cc];
        p_avg = new float[n_cc];
        p_std = new float[n_cc];
        p_cnt = new float[n_cc];
        sigma = in_sigma;

        p_x = new int[n_pts];
        p_y = new int[n_pts];
        p_z = new int[n_pts];

        for(int i=0;i<n_pts;i++) {
            p_x[i] = (int)roundf(c_pts[i].x);
            p_y[i] = (int)roundf(c_pts[i].y);
            p_z[i] = (int)roundf(c_pts[i].z);
        }
    }

    ~TemplateMatchingReporter() {
        delete [] c_cc;
        delete [] p_avg;
        delete [] p_std;
        delete [] p_cnt;
        delete [] p_x;
        delete [] p_y;
        delete [] p_z;
    }

    void start(int id,const char*type,const char*prefix) {
        if( strcmp(type,"none") == 0 ) {
            tm_type = TM_NONE;
        }
        else if( strcmp(type,"python") == 0 ) {
            tm_type = TM_PYTHON;
        }
        else if( strcmp(type,"matlab") == 0 ) {
            tm_type = TM_MATLAB;
        }
        else if( strcmp(type,"csv") == 0 ) {
            tm_type = TM_CSV;
        }

        if( tm_type == TM_PYTHON || tm_type == TM_MATLAB || tm_type == TM_CSV ) {
            char tm_file[SUSAN_FILENAME_LENGTH];
            sprintf(tm_file,"%s_worker%02d.txt",prefix,id);
            fp = fopen(tm_file,"w");
            if( tm_type == TM_CSV ) {
                if (tm_dim == 2){
                    fprintf(fp,"TID,PartID,RID,ProjID,ProjW,X,Y,CC\n");
                }
                else if (tm_dim == 3){
                    fprintf(fp,"TID,PartID,RID,X,Y,Z,CC,BlockID\n");
                    block_id = 0;
                }
            }
        }
    }

    void finish() {
        if( tm_type == TM_PYTHON || tm_type == TM_MATLAB || tm_type == TM_CSV ) {
            fclose(fp);
        }
    }

    void clear_cc() {
        memset(c_cc ,0,n_cc*sizeof(float));
        memset(p_avg,0,n_cc*sizeof(float));
        memset(p_std,0,n_cc*sizeof(float));
        memset(p_cnt,0,n_cc*sizeof(float));
    }

   void push_cc(const float*p_cc) {
        if( tm_type == TM_NONE )
            return;

        for(int cc_index=0;cc_index<n_cc;cc_index++) {
            float cc = p_cc[cc_index];
            c_cc [cc_index] = cc;
            p_avg[cc_index] += cc;
            p_std[cc_index] += (cc*cc);
            p_cnt[cc_index] += 1;
        }
    }

    void save_cc(int tid,int rid,int pid,int tx,int ty,int tz,float *prj_w,bool save_sigma=false) {
        if( tm_type == TM_NONE )
            return;

        int x,y,z;

        int proj_id, point_id;
        float proj_w;

        for(int cc_index=0;cc_index<n_cc;cc_index++){
            if( p_cnt[cc_index] == 0 )
                continue;

            if (tm_dim == 2){
                proj_id  = cc_index / num_points;
                point_id = cc_index % num_points;
                proj_w   = prj_w[proj_id];
            }
            else if (tm_dim == 3){
                proj_id  = 0;
                point_id = cc_index;
            }

            x = p_x[point_id];
            y = p_y[point_id];
            z = p_z[point_id];

            if(save_sigma) {
                float cc_avg = p_avg[cc_index]/p_cnt[cc_index];
                float cc_std = p_std[cc_index]/p_cnt[cc_index];
                cc_std = cc_std - (cc_avg*cc_avg);
                cc_std = sqrtf(cc_std);
                c_cc[cc_index] = (c_cc[cc_index]-cc_avg)/cc_std;
            }

	    if (tm_dim == 2){
	        if( tm_type == TM_PYTHON )
		        fprintf(fp,"cc_tomo%03d_ptcl%d_ref%02d_proj%02d_w%f[%d,%d] = %.15lf\n", tid, pid, rid, proj_id, proj_w, x, y, c_cc[cc_index]);
	        else if( tm_type == TM_MATLAB )
		        fprintf(fp,"cc_tomo%03d_ptcl%d_ref%02d_proj%02d_w%.15lf(%d,%d) = %f;\n",tid, pid, rid, proj_id, proj_w, (x+1), (y+1), c_cc[cc_index]);
	        else if( tm_type == TM_CSV )
		        fprintf(fp,"%d,%d,%d,%d,%f,%d,%d,%.15lf\n", tid, pid, rid, proj_id, proj_w, x, y, c_cc[cc_index]);
	    }
	    else if ((tm_dim == 3) && (c_cc[cc_index] > 0)){
	        if( tm_type == TM_PYTHON )
		        fprintf(fp,"cc_tomo%03d_ptcl%d_ref%02d[%d,%d,%d] = %.15lf\n", tid, pid, rid, (z+tz),  (y+ty),  (x+tx),  c_cc[cc_index]);
	        else if( tm_type == TM_MATLAB )
		        fprintf(fp,"cc_tomo%03d_ptcl%d_ref%02d(%4d,%4d,%4d) = %.15lf;\n",tid, pid, rid, (x+tx+1), (y+ty+1), (z+tz+1), c_cc[cc_index]);
	        else if( tm_type == TM_CSV ) {
		        fprintf(fp,"%d,%d,%d,%d,%d,%d,%.15lf,%d\n",tid, pid, rid, (x+tx), (y+ty), (z+tz), c_cc[cc_index],block_id);
	        }
	    }
	}
        block_id++;
    }

};

class CcStatsTracker {

protected:
    single running_sum;
    single running_sqsum;
    int    running_count;

    single current_cc;

    Vec3 current_vec;
    M33f current_rot;
    Eigen::Matrix4f weighted_rotation;
    int type;

protected:
    static void get_max_argmax_sum_sqsum(single&max_val,int&max_idx,single&sum_val,single&sqsum_val,const single*p_data,const int n_data,bool allow_negative=false) {
        max_idx = 0;
        max_val = p_data[max_idx];
        sum_val = 0;
        sqsum_val = 0;
        if(!allow_negative){
            max_val = fmax(max_val,0.0);
        }
        for(int i=0;i<n_data;i++) {
            float cur_val = p_data[i];
            if(!allow_negative){
                cur_val = fmax(cur_val,0.0);
            }
            sum_val += cur_val;
            sqsum_val += (cur_val*cur_val);
            if( max_val < cur_val ) {
                max_val = cur_val;
                max_idx = i;
            }
        }
    }

    bool check_stats_failed() {
        if( type == CC_STATS_NONE ) {
            return isinf(current_cc) || (running_sum==0);
        }

        if( type == CC_STATS_PROB ) {
            return isinf(current_cc) || (current_cc>running_sum) || (running_sum==0);
        }

        if( type == CC_STATS_SIGMA ) {
            return isinf(current_cc) || (current_cc>running_sum) || (running_count<1) || (running_sum==0);
        }

        if( type == CC_STATS_WGT_AVG ) {
            return isinf(current_cc) || (current_cc>running_sum) || (running_count<1) || (running_sum==0);
        }

        return isinf(current_cc) || (current_cc>running_sum) || (running_count<1) || (running_sum==0);
    }

public:
    CcStatsTracker(int cc_stats_type=CC_STATS_NONE) {
        type = cc_stats_type;
        clear();
    }

    ~CcStatsTracker() {
    }

    void clear() {
        current_cc  = -INFINITY;
        running_sum = 0;
        running_sqsum = 0;
        running_count = 0;
        current_rot = M33f::Identity();
        current_vec.x = 0;
        current_vec.y = 0;
        current_vec.z = 0;
        weighted_rotation = Eigen::Matrix4f::Zero();

    }

    void push( const Vec3*p_pts,const float*p_cc,const int n_pts,const M33f&Rot) {
        single max_val;
        int    max_idx;
        single sum_val;
        single sqsum_val;
        get_max_argmax_sum_sqsum(max_val,max_idx,sum_val,sqsum_val,p_cc,n_pts,type == CC_STATS_NONE);

        if( type == CC_STATS_NONE ) {
            if( current_cc < max_val ) {
                current_cc    = max_val;
                current_vec.x = p_pts[max_idx].x;
                current_vec.y = p_pts[max_idx].y;
                current_vec.z = p_pts[max_idx].z;
                current_rot   = Rot;
            }
        }

        if( type == CC_STATS_PROB ) {
            if( current_cc < max_val ) {
                current_cc     = max_val;
                current_vec.x  = p_pts[max_idx].x;
                current_vec.y  = p_pts[max_idx].y;
                current_vec.z  = p_pts[max_idx].z;
                current_rot    = Rot;
            }
            running_sum   += sum_val;
        }

        if( type == CC_STATS_SIGMA ) {
            if( current_cc < max_val ) {
                current_cc     = max_val;
                current_vec.x  = p_pts[max_idx].x;
                current_vec.y  = p_pts[max_idx].y;
                current_vec.z  = p_pts[max_idx].z;
                current_rot    = Rot;
            }
            running_sum   += sum_val;
            running_sqsum += sqsum_val;
            running_count += n_pts;
        }

        if( type == CC_STATS_WGT_AVG ) {
            if( current_cc  < max_val ) {
                current_cc  = max_val;
                current_rot = Rot;
            }
            running_sum   += sum_val;
            running_sqsum += sqsum_val;
            running_count += n_pts;

            for(int i=0;i<n_pts;i++) {
                single wgt = fmax(p_cc[i],0.0);
                current_vec.x += wgt*p_pts[i].x;
                current_vec.y += wgt*p_pts[i].y;
                current_vec.z += wgt*p_pts[i].z;
            }

            Eigen::Quaternionf quat(Rot);
            Eigen::Vector4f q;
            if( quat.w() < 0 )
                q = -quat.coeffs();
            else
                q = quat.coeffs();
            weighted_rotation = q*q.adjoint()*sum_val + weighted_rotation;
        }
    }

    single get_cc() {
        if( check_stats_failed() )
            return 0;

        if( type == CC_STATS_NONE )
            return current_cc;

        if( (type == CC_STATS_PROB) || (type==CC_STATS_WGT_AVG) ){
            return current_cc / running_sum;
        }

        if( type==CC_STATS_SIGMA ) {
            single avg = running_sum   / running_count;
            single std = running_sqsum / running_count;
            std = std - (avg*avg);
            std = sqrtf( fmax( std, 0.0 ) );
            if( std <= 0 )
                return 0;
            return fmax( (current_cc - avg) / std, 0.0 );
        }

        return 0;
    }

    Vec3 get_vec() {
        Vec3 out = current_vec;

        if( check_stats_failed() ) {
            out.x = 0;
            out.y = 0;
            out.z = 0;
            return out;
        }

        if( (type==CC_STATS_NONE) || (type==CC_STATS_PROB) || (type==CC_STATS_SIGMA) )
            return out;

        if( type == CC_STATS_WGT_AVG ) {
            out.x = current_vec.x / running_sum;
            out.y = current_vec.y / running_sum;
            out.z = current_vec.z / running_sum;
            return out;
        }
        return out;
    }

    M33f get_rot() {
        if( check_stats_failed() ) {
            return Eigen::MatrixXf::Identity(3,3);
        }

        if( (type==CC_STATS_NONE) || (type==CC_STATS_PROB) || (type==CC_STATS_SIGMA) )
            return current_rot;

        if( type == CC_STATS_WGT_AVG ) {
            Eigen::Matrix4f W = (1.0f/running_sum) * weighted_rotation;
            Eigen::SelfAdjointEigenSolver<Matrix4f> eig(W);
            Eigen::Vector4f q = eig.eigenvectors().col(3);
            Eigen::Quaternionf quat(q);
            return quat.toRotationMatrix();
        }

        return current_rot;
    }

    M33f get_lvl_rot() {
        return current_rot;
    }

    void print_stats() {
        if( type == CC_STATS_NONE ) {
            printf("CC: %e\n",current_cc);
        }

        if( type == CC_STATS_PROB ) {
            printf("CC: %e\nsum_cc: %e\n",current_cc,running_sum);
        }

        if( type == CC_STATS_SIGMA ) {
            printf("CC: %e\nsum_cc: %e\n",current_cc,running_sum);
            printf("sqsum: %e\ncount: %.0f\n",running_sqsum,running_count);

        }

        if( type == CC_STATS_WGT_AVG ) {
            printf("WGT_CC: %e\nsum_cc: %e\n",current_cc,running_sum);
            printf("sqsum: %e\ncount: %.0f\n",running_sqsum,running_count);
        }
    }
};

class CcStatsTrackerArr {

public:
    CcStatsTracker**cc_stats_arr;
protected:
    int numel;

public:
    CcStatsTrackerArr(int cc_stats_type,const int k) {
        numel = k;
        cc_stats_arr = new CcStatsTracker*[numel];
        for(int i=0;i<numel;i++) {
            cc_stats_arr[i] = new CcStatsTracker(cc_stats_type);
        }
    }

    ~CcStatsTrackerArr() {
        for(int i=0;i<numel;i++) {
            delete cc_stats_arr[i];
        }
        delete [] cc_stats_arr;
    }

    void clear() {
        for(int i=0;i<numel;i++) {
            cc_stats_arr[i]->clear();
        }
    }

    void push( const Vec3*p_pts,const float*p_cc,const int n_pts,const M33f&Rot) {
        for(int i=0;i<numel;i++) {
            int off = i*n_pts;
            cc_stats_arr[i]->push(p_pts,p_cc+off,n_pts,Rot);
        }
    }

    single get_cc(int idx) {
        return cc_stats_arr[idx]->get_cc();
    }

    Vec3 get_vec(int idx) {
        return cc_stats_arr[idx]->get_vec();
    }

    M33f get_rot(int idx) {
        return cc_stats_arr[idx]->get_rot();
    }
};

class AliGpuWorker : public Worker {

public:
    int gpu_ix;
    int N;
    int M;
    int P;
    int R;
    int pad_type;
    int ctf_type;
    int cc_type;
    int cc_stats;
    int max_K;
    int dilate;
    bool ali_halves;
    float3  bandpass;
    float2  ssnr; /// x=F; y=S;
    DoubleBufferHandler *p_buffer;
    RefMap              *p_refs;

    const char*psym;
    float2 cone; /// x=range; y=step
    float2 inplane; /// x=range; y=step
    uint32 ref_level;
    uint32 ref_factor;
    uint32 off_type;
    uint32 off_space;
    float4 off_par;

    bool drift2D;
    bool drift3D;

    AnglesProvider ang_prov;
    
    const char *tm_type;
    const char *tm_prefix;
    int         tm_dim;
    float       tm_sigma;

    AliGpuWorker() {
    }

    ~AliGpuWorker() {
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

        AliData ali_data(MP,NP,max_K,off_par,off_type,stream);
        
        TemplateMatchingReporter tm_rep(ali_data.c_pts,ali_data.n_pts,max_K,tm_dim,tm_sigma);
        tm_rep.start(worker_id,tm_type,tm_prefix);

        RadialAverager rad_avgr(MP,NP,max_K);

        GPU::GArrSingle ctf_wgt;
        ctf_wgt.alloc(MP*NP*max_K);

        int num_vols = R;
        if( ali_halves ) num_vols = 2*R;
        AliRef*vols = new AliRef[num_vols];
        allocate_references(vols,rad_avgr);

        ang_prov.cone_range = cone.x;
        ang_prov.cone_step  = cone.y;
        ang_prov.inplane_range = inplane.x;
        ang_prov.inplane_step  = inplane.y;
        ang_prov.refine_factor = ref_factor;
        ang_prov.refine_level  = ref_level;
        ang_prov.set_symmetry(psym);

        GPU::sync();

        while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
            switch(current_cmd) {
                case ALI_3D:
                    align3D(vols,ctf_wgt,ss_data,ali_data,rad_avgr,tm_rep,stream);
                    break;
                case ALI_2D:
                    align2D(vols,ctf_wgt,ss_data,ali_data,rad_avgr,tm_rep,stream);
                    break;
                default:
                    break;
            }
        }

        GPU::sync();
        tm_rep.finish();
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

        if( ali_halves ) {
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
        dim3 blk  = GPU::get_block_size_2D();
        dim3 grdR = GPU::calc_grid_size(blk,NP,NP,NP);
        dim3 grdC = GPU::calc_grid_size(blk,MP,NP,NP);
        int3 ss   = make_int3(MP,NP,NP);
        GpuKernels::fftshift3D<<<grdR,blk>>>(g_pad.ptr,NP);
        fft3.exec(g_fou.ptr,g_pad.ptr);
        GpuKernels::sampling_correction_3D<<<grdC,blk>>>(g_fou.ptr,2.0,MP,NP);
        GpuKernels::fftshift3D<<<grdC,blk>>>(g_fou.ptr,MP,NP);
        GpuKernels::divide<<<grdC,blk>>>(g_fou.ptr,NP*NP*NP,ss);
    }

    void align3D(AliRef*vols,GPU::GArrSingle&ctf_wgt,AliSubstack&ss_data,AliData&ali_data,RadialAverager&rad_avgr,TemplateMatchingReporter&tm_rep,GPU::Stream&stream) {
        p_buffer->RO_sync();
        while( p_buffer->RO_get_status() > DONE ) {
            if( p_buffer->RO_get_status() == READY ) {
                AliBuffer*ptr = (AliBuffer*)p_buffer->RO_get_buffer();
                create_ctf(ctf_wgt,ptr,rad_avgr,stream);
                add_data(ss_data,ctf_wgt,ptr,rad_avgr,stream);
                add_rec_weight(ss_data,ptr,stream);
                angular_search_3D(vols[ptr->r_ix],ss_data,ctf_wgt,ptr,ali_data,rad_avgr,tm_rep,stream);
                stream.sync();
            }
            p_buffer->RO_sync();
        }
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

    void debug_ctf_stack(const char*filename,GPU::GArrSingle&g_real,GPU::Stream&stream) {
        GPU::GHostSingle buffer;
        buffer.alloc(MP*NP*max_K);
        GPU::download_async(buffer.ptr,g_real.ptr,MP*NP*max_K,stream.strm);
        stream.sync();
        Mrc::write(buffer.ptr,MP,NP,max_K,filename);
    }

    void align2D(AliRef*vols,GPU::GArrSingle&ctf_wgt,AliSubstack&ss_data,AliData&ali_data,RadialAverager&rad_avgr,TemplateMatchingReporter&tm_rep,GPU::Stream&stream) {
        p_buffer->RO_sync();
        while( p_buffer->RO_get_status() > DONE ) {
            if( p_buffer->RO_get_status() == READY ) {
                AliBuffer*ptr = (AliBuffer*)p_buffer->RO_get_buffer();
                create_ctf(ctf_wgt,ptr,rad_avgr,stream);
                add_data(ss_data,ctf_wgt,ptr,rad_avgr,stream);
                angular_search_2D(vols[ptr->r_ix],ss_data,ctf_wgt,ptr,ali_data,rad_avgr,tm_rep,stream);
                stream.sync();
            }
            p_buffer->RO_sync();
        }
    }

    void create_ctf(GPU::GArrSingle&ctf_wgt,AliBuffer*ptr,RadialAverager&rad_avgr,GPU::Stream&stream) {
        int3 ss = make_int3(MP,NP,ptr->K);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,MP,NP,ptr->K);
        GpuKernelsCtf::create_ctf<<<grd,blk,0,stream.strm>>>(ctf_wgt.ptr,ptr->ctf_vals,ptr->g_def.ptr,false,ss);
        rad_avgr.normalize_ctf(ctf_wgt,ptr->K,stream);
    }

    void add_data(AliSubstack&ss_data,GPU::GArrSingle&ctf_wgt,AliBuffer*ptr,RadialAverager&rad_avgr,GPU::Stream&stream) {

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

        switch( ctf_type ) {
            case ALI_CTF_ON_SUBSTACK:
                ss_data.correct_wiener(ptr->ctf_vals,ctf_wgt,ptr->g_def,bandpass,ptr->K,stream);
                break;
            case ALI_CTF_ON_SUBSTACK_SSNR:
                ss_data.correct_wiener_ssnr(ptr->ctf_vals,ctf_wgt,ptr->g_def,bandpass,ssnr,ptr->K,stream);
                break;
            default:
                break;
        }

        if( cc_type == CC_TYPE_CFSC ) {
            rad_avgr.calculate_FRC(ss_data.ss_fourier,ptr->K,stream);
            rad_avgr.apply_FRC(ss_data.ss_fourier,ptr->ctf_vals,ssnr,ptr->K,stream);
        }
    }

    void add_rec_weight(AliSubstack&ss_data,AliBuffer*ptr,GPU::Stream&stream) {
        float w_total=0;
        for(int k=0;k<ptr->K;k++) {
            w_total += ptr->c_ali.ptr[k].w;
        }
        ss_data.apply_radial_wgt(w_total,ptr->crowther_limit,ptr->K,stream);
    }

    void print_R(GPU::GArrProj2D&g_ali,int k,GPU::Stream&stream) {
        dim3 blk;
        dim3 grd;
        blk.x = 1024;
        blk.y = 1;
        blk.z = 1;
        grd.x = GPU::div_round_up(9*k,1024);
        grd.y = 1;
        grd.z = 1;
        GpuKernels::print_proj2D<<<grd,blk,0,stream.strm>>>(g_ali.ptr,k);
        stream.sync();
    }

    void angular_search_3D(AliRef&vol,AliSubstack&ss_data,GPU::GArrSingle&ctf_wgt,AliBuffer*ptr,AliData&ali_data,RadialAverager&rad_avgr,TemplateMatchingReporter&tm_rep,GPU::Stream&stream) {

        M33f max_R;

        Rot33 Rot;
        M33f R_lvl = Eigen::MatrixXf::Identity(3,3);
        M33f R_ite,R_tmp,R_ali;

        CcStatsTracker cc_tracker(cc_stats);

        Math::eZYZ_Rmat(R_ali,ptr->ptcl.ali_eu[ptr->class_ix]);

        tm_rep.clear_cc();

        // DEBUG
        // if( ptr->ptcl.ptcl_id() == 0 )
            // debug_fourier_stack("sus.mrc",ss_data.ss_fourier,stream);

        /*if( ptr->ptcl.ptcl_id() == 104 ) {
            printf("\n\nR_ali = np.zeros((3,3)) #%d\n",ptr->K);
            printf("R_ali[0,:] = (%f,%f,%f)\n",R_ali(0,0),R_ali(0,1),R_ali(0,2));
            printf("R_ali[1,:] = (%f,%f,%f)\n",R_ali(1,0),R_ali(1,1),R_ali(1,2));
            printf("R_ali[2,:] = (%f,%f,%f)\n",R_ali(2,0),R_ali(2,1),R_ali(2,2));
            //print_R(ptr->g_ali,ptr->K,stream);
        }*/

        for( ang_prov.levels_init(); ang_prov.levels_available(); ang_prov.levels_next() ) {
            //if( ptr->ptcl.ptcl_id() == 2 ) printf("Level ======\n");
            for( ang_prov.sym_init(); ang_prov.sym_available(); ang_prov.sym_next() ) {
                for( ang_prov.cone_init(); ang_prov.cone_available(); ang_prov.cone_next() ) {
                    for( ang_prov.inplane_init(); ang_prov.inplane_available(); ang_prov.inplane_next() ) {

                        //if( ptr->ptcl.ptcl_id() == 2 ) ang_prov.print_curr_angles();

                        ang_prov.get_current_R(R_ite);
                        R_tmp = (R_ite*R_lvl*R_ali).transpose();
                        Math::set(Rot,R_tmp);

                        ali_data.rotate_reference(Rot,ptr->g_ali,ptr->K,stream);
                        ali_data.project(vol.ref,bandpass,ptr->K,stream);

                        /*if( ptr->ptcl.ptcl_id() == 0 ) {
                            stream.sync();
                            printf("\nR_ite = np.zeros((3,3)) #%d\n",ptr->K);
                            printf("R_ite[0,:] = (%f,%f,%f)\n",R_tmp(0,0),R_tmp(0,1),R_tmp(0,2));
                            printf("R_ite[1,:] = (%f,%f,%f)\n",R_tmp(1,0),R_tmp(1,1),R_tmp(1,2));
                            printf("R_ite[2,:] = (%f,%f,%f)\n",R_tmp(2,0),R_tmp(2,1),R_tmp(2,2));
                            //print_R(ali_data.g_ali,ptr->K,stream);
                        }*/

                        if( cc_type == CC_TYPE_CFSC ) {
                            rad_avgr.calculate_FRC(ali_data.prj_c,ptr->K,stream);
                            rad_avgr.apply_FRC(ali_data.prj_c,ptr->K,stream);
                        }

                        if( ctf_type == ALI_CTF_ON_REFERENCE )
                            ali_data.multiply(ctf_wgt,ptr->K,stream);

                        // if( cc_type == CC_TYPE_CFSC )
                        //     rad_avgr.apply_FRC(ali_data.prj_c,ptr->K,stream);

                        // DEBUG
                        // if( ptr->ptcl.ptcl_id() == 0 )
                            // debug_fourier_stack("prj.mrc",ali_data.prj_c,stream);

                        ali_data.apply_bandpass(ptr->ctf_vals,ptr->g_def,bandpass,ptr->K,stream);
                        ali_data.multiply(ss_data.ss_fourier,ptr->K,stream);

                        // DEBUG
                        // debug_fourier_stack("cc.mrc",ali_data.prj_c,stream);
                        // if( ptr->ptcl.ptcl_id() == 0 )
                        //     debug_fourier_stack("cc.mrc",ali_data.prj_c,stream);

                        ali_data.invert_fourier(ptr->K,stream);

                        ali_data.sparse_reconstruct(ptr->g_ali,dilate,ptr->K,stream);
                        stream.sync();

                        cc_tracker.push(ali_data.c_pts,ali_data.c_cc,ali_data.n_pts,R_ite*R_lvl);

                        //if( ptr->ptcl.ptcl_id() == 2 ) printf("cc = %f\n",cc);

                        tm_rep.push_cc(ali_data.c_cc);
                    } // INPLANE
                } // CONE
            } // SYMMETRY
            //R_lvl = max_R;
            R_lvl = cc_tracker.get_lvl_rot();
        } // REFINE

        update_particle_3D( ptr->ptcl,
                           cc_tracker.get_rot(),cc_tracker.get_vec(),cc_tracker.get_cc(),
                           ptr->class_ix,ptr->ctf_vals.apix);
        tm_rep.save_cc(ptr->ptcl.tomo_id(),ptr->ptcl.ref_cix()+1,ptr->ptcl.ptcl_id(),ptr->tomo_pos_x,ptr->tomo_pos_y,ptr->tomo_pos_z,ptr->ptcl.prj_w,cc_stats==CC_STATS_SIGMA);
    }

    void angular_search_2D(AliRef&vol,AliSubstack&ss_data,GPU::GArrSingle&ctf_wgt,AliBuffer*ptr,AliData&ali_data,RadialAverager&rad_avgr,TemplateMatchingReporter&tm_rep,GPU::Stream&stream) {
        
        Rot33 Rot;
        M33f  R_ite,R_ali;

        CcStatsTrackerArr cc_tracker_arr(cc_stats,ptr->K);

        single max_cc [ptr->K];
        single ite_cc [ptr->K];
        int    max_idx[ptr->K];
        int    ite_idx[ptr->K];
        M33f   max_R  [ptr->K];
        // single := float //
        single cc_placeholder[(ptr->K)*(ali_data.n_pts)];

        Math::eZYZ_Rmat(R_ali,ptr->ptcl.ali_eu[ptr->class_ix]);

        for(int i=0;i<ptr->K;i++) max_cc[i] = -INFINITY;
        memset(max_idx,        0, sizeof(single)*ptr->K);
        memset(cc_placeholder, 0, sizeof(single)*(ptr->K)*(ali_data.n_pts));

        tm_rep.clear_cc();

        // DEBUG
        // int it = 0;
        /*long int cur_time = time(NULL);
        char name[2048];
        if( ptr->ptcl.ptcl_id() == 295 ) {
            sprintf(name,"sus_%ld.mrc",cur_time);
            debug_fourier_stack(name,ss_data.ss_fourier,stream);
        }*/
        // if( ptr->ptcl.ptcl_id() == 0 )
            // debug_fourier_stack("sus.mrc",ss_data.ss_fourier,stream);

        Math::set(Rot,R_ali.transpose());
        ali_data.pre_rotate_reference(Rot,ptr->g_ali,ptr->K,stream);

        //if( ptr->ptcl.ptcl_id() == 44 ) {
            // printf("\n\nR_ali = np.zeros((3,3)) #%d\n",ptr->K);
            // printf("R_ali[0,:] = (%f,%f,%f)\n",R_ali(0,0),R_ali(0,1),R_ali(0,2));
            // printf("R_ali[1,:] = (%f,%f,%f)\n",R_ali(1,0),R_ali(1,1),R_ali(1,2));
            // printf("R_ali[2,:] = (%f,%f,%f)\n",R_ali(2,0),R_ali(2,1),R_ali(2,2));
            // print_R(ptr->g_ali,ptr->K,stream);
        //}

        for( ang_prov.levels_init(); ang_prov.levels_available(); ang_prov.levels_next() ) {
            for( ang_prov.sym_init(); ang_prov.sym_available(); ang_prov.sym_next() ) {
                for( ang_prov.cone_init(); ang_prov.cone_available(); ang_prov.cone_next() ) {
                    for( ang_prov.inplane_init(); ang_prov.inplane_available(); ang_prov.inplane_next() ) {

                        ang_prov.get_current_R(R_ite);
                        Math::set(Rot,R_ite.transpose());

                        ali_data.rotate_projections(Rot,ptr->g_ali,ptr->K,stream);
                        ali_data.project(vol.ref,bandpass,ptr->K,stream);

                        //if( ptr->ptcl.ptcl_id() == 44 ) {
                            // stream.sync();
                            // printf("\nR_ite = np.zeros((3,3)) #%d\n",ptr->K);
                            // printf("R_ite[0,:] = (%f,%f,%f)\n",R_ite(0,0),R_ite(0,1),R_ite(0,2));
                            // printf("R_ite[1,:] = (%f,%f,%f)\n",R_ite(1,0),R_ite(1,1),R_ite(1,2));
                            // printf("R_ite[2,:] = (%f,%f,%f)\n",R_ite(2,0),R_ite(2,1),R_ite(2,2));
                            // print_R(ali_data.g_ali,ptr->K,stream);
                        //}

                        if( cc_type == CC_TYPE_CFSC ) {
                            rad_avgr.calculate_FRC(ali_data.prj_c,ptr->K,stream);
                            rad_avgr.apply_FRC(ali_data.prj_c,ptr->K,stream);
                        }

                        if( ctf_type == ALI_CTF_ON_REFERENCE )
                            ali_data.multiply(ctf_wgt,ptr->K,stream);

                        // DEBUG
                        /*if( ptr->ptcl.ptcl_id() == 295 ) {
                            sprintf(name,"prj_%ld_%03d.mrc",cur_time,it);
                            debug_fourier_stack(name,ali_data.prj_c,stream);
                        }*/
                        // if( ptr->ptcl.ptcl_id() == 0 )
                            // debug_fourier_stack("prj.mrc",ali_data.prj_c,stream);

                        ali_data.apply_bandpass(ptr->ctf_vals,ptr->g_def,bandpass,ptr->K,stream);
                        ali_data.multiply(ss_data.ss_fourier,ptr->K,stream);

                        // DEBUG
                        /*if( ptr->ptcl.ptcl_id() == 295 ) {
                            sprintf(name,"cc_%ld_%03d.mrc",cur_time,it);
                            debug_fourier_stack(name,ali_data.prj_c,stream);
                        }*/
                        // if( ptr->ptcl.ptcl_id() == 0 )
                            // debug_fourier_stack("cc.mrc",ali_data.prj_c,stream);

                        ali_data.invert_fourier(ptr->K,stream);
                        stream.sync();

                        ali_data.extract_cc(ite_cc,ite_idx,ptr->g_ali,ptr->K,stream);

                        cc_tracker_arr.push(ali_data.c_pts,ali_data.c_cc,ali_data.n_pts,R_ite);

                        /*if( ptr->ptcl.ptcl_id() == 295 ) {
                            printf("\n\nCC_ARR = np.array((\n");
                            for(int j=0;j<ali_data.n_pts;j++){
                                printf("\t%e,\n",ali_data.c_cc[(14*ali_data.n_pts)+j]);
                            }
                            printf("))\n");
                            cc_tracker_arr.cc_stats_arr[14]->print_stats();
                        }*/

                        for(int i=0;i<ptr->K;i++) {
                            if( ite_cc[i] > max_cc[i] ) {
                                max_idx[i] = ite_idx[i];
                                max_cc[i]  = ite_cc[i];
                                max_R[i]   = R_ite;
                                // cc_placeholder stores cc-map (offset->cc) only for the best orientation
                                for(int j=0;j<ali_data.n_pts;j++){
                                    cc_placeholder[i*ali_data.n_pts + j] = ali_data.c_cc[i*ali_data.n_pts + j];
                                }
                                // if( i == 30 ) {
                                //     //if( ptr->ptcl.ptcl_id() == 44 ) {
                                //         stream.sync();
                                //         printf("\nR_ite = np.zeros((3,3)) #%d\n",ptr->K);
                                //         printf("R_ite[0,:] = (%f,%f,%f)\n",R_ite(0,0),R_ite(0,1),R_ite(0,2));
                                //         printf("R_ite[1,:] = (%f,%f,%f)\n",R_ite(1,0),R_ite(1,1),R_ite(1,2));
                                //         printf("R_ite[2,:] = (%f,%f,%f)\n",R_ite(2,0),R_ite(2,1),R_ite(2,2));
                                //         printf("cc = %f; it = %d\n",ite_cc[i],it);
                                //         print_R(ali_data.g_ali,ptr->K,stream);
                                //     //}
                                // }
                            }
                        }
                        // it++;
                    } // INPLANE
                } // CONE
            } // SYMMETRY
        } // REFINE

        tm_rep.push_cc(cc_placeholder);

        single cc_acc=0,wgt_acc=0,cc_cur=0;
        for(int i=0;i<ptr->K;i++) {
            if( ptr->ptcl.prj_w[i] > 0 ) {
                cc_cur  = cc_tracker_arr.get_cc(i);
                cc_acc += cc_cur;
                wgt_acc += ptr->ptcl.prj_w[i];
                update_particle_2D(ptr->ptcl,
                                   cc_tracker_arr.get_rot(i),cc_tracker_arr.get_vec(i),cc_cur,
                                   i,ptr->ctf_vals.apix);
            }
        }
        ptr->ptcl.ali_cc[ptr->class_ix] = cc_acc/fmax(wgt_acc,1.0);
        tm_rep.save_cc(ptr->ptcl.tomo_id(),ptr->ptcl.ref_cix()+1,ptr->ptcl.ptcl_id(),ptr->tomo_pos_x,ptr->tomo_pos_y,ptr->tomo_pos_z,ptr->ptcl.prj_w);
    }

    void update_particle_3D(Particle&ptcl,const M33f&Rot,const Vec3&t,const single cc, const int ref_ix,const float apix) {

        ptcl.ali_cc[ref_ix] = cc;

        M33f Rprv;
        Math::eZYZ_Rmat(Rprv,ptcl.ali_eu[ref_ix]);
        //M33f Rnew = Rprv*Rot;
        M33f Rnew = Rot*Rprv;

        /*if( ptcl.ptcl_id() == 104 ) {
            printf("\nRSLT ========\n");
            printf("\n Rprv = np.zeros((3,3))\n");
            printf("Rprv[0,:] = (%f,%f,%f)\n",Rprv(0,0),Rprv(0,1),Rprv(0,2));
            printf("Rprv[1,:] = (%f,%f,%f)\n",Rprv(1,0),Rprv(1,1),Rprv(1,2));
            printf("Rprv[2,:] = (%f,%f,%f)\n",Rprv(2,0),Rprv(2,1),Rprv(2,2));
            printf("\n Rot = np.zeros((3,3))\n");
            printf("Rot[0,:] = (%f,%f,%f)\n",Rot(0,0),Rot(0,1),Rot(0,2));
            printf("Rot[1,:] = (%f,%f,%f)\n",Rot(1,0),Rot(1,1),Rot(1,2));
            printf("Rot[2,:] = (%f,%f,%f)\n",Rot(2,0),Rot(2,1),Rot(2,2));
            printf("\n Rnew = np.zeros((3,3))\n");
            printf("Rnew[0,:] = (%f,%f,%f)\n",Rnew(0,0),Rnew(0,1),Rnew(0,2));
            printf("Rnew[1,:] = (%f,%f,%f)\n",Rnew(1,0),Rnew(1,1),Rnew(1,2));
            printf("Rnew[2,:] = (%f,%f,%f)\n",Rnew(2,0),Rnew(2,1),Rnew(2,2));
            printf("cc = %f\n",cc);
        }*/

        Math::Rmat_eZYZ(ptcl.ali_eu[ref_ix],Rnew);
        /*if( ptcl.ptcl_id() == 104 ) {
            printf("euler = %f, %f, %f\n",ptcl.ali_eu[ref_ix].x,ptcl.ali_eu[ref_ix].y,ptcl.ali_eu[ref_ix].z);
        }*/

        if( drift3D ) {
            ptcl.ali_t[ref_ix].x += t.x*apix;
            ptcl.ali_t[ref_ix].y += t.y*apix;
            ptcl.ali_t[ref_ix].z += t.z*apix;
        }
        else {
            ptcl.ali_t[ref_ix].x = t.x*apix;
            ptcl.ali_t[ref_ix].y = t.y*apix;
            ptcl.ali_t[ref_ix].z = t.z*apix;
        }
    }

    void update_particle_2D(Particle&ptcl,const M33f&Rot,const Vec3&t,const single cc, const int prj_ix,const float apix) {

        if( ptcl.prj_w[prj_ix] > 0 ) {
            ptcl.prj_cc[prj_ix] = cc;

            M33f Rprv;
            Math::eZYZ_Rmat(Rprv,ptcl.prj_eu[prj_ix]);
            M33f Rnew = Rot*Rprv;
            Math::Rmat_eZYZ(ptcl.prj_eu[prj_ix],Rnew);

            /*if( ptcl.ptcl_id() == 44 && prj_ix == 20 ) {
                printf("\nRprv = np.zeros((3,3))\n");
                printf("Rprv[0,:] = (%f,%f,%f)\n",Rprv(0,0),Rprv(0,1),Rprv(0,2));
                printf("Rprv[1,:] = (%f,%f,%f)\n",Rprv(1,0),Rprv(1,1),Rprv(1,2));
                printf("Rprv[2,:] = (%f,%f,%f)\n",Rprv(2,0),Rprv(2,1),Rprv(2,2));
                printf("\nRot = np.zeros((3,3))\n");
                printf("Rot[0,:] = (%f,%f,%f)\n",Rot(0,0),Rot(0,1),Rot(0,2));
                printf("Rot[1,:] = (%f,%f,%f)\n",Rot(1,0),Rot(1,1),Rot(1,2));
                printf("Rot[2,:] = (%f,%f,%f)\n",Rot(2,0),Rot(2,1),Rot(2,2));
                printf("\nRnew = np.zeros((3,3))\n");
                printf("Rnew[0,:] = (%f,%f,%f)\n",Rnew(0,0),Rnew(0,1),Rnew(0,2));
                printf("Rnew[1,:] = (%f,%f,%f)\n",Rnew(1,0),Rnew(1,1),Rnew(1,2));
                printf("Rnew[2,:] = (%f,%f,%f)\n",Rnew(2,0),Rnew(2,1),Rnew(2,2));
            }*/

            if( drift2D ) {
                ptcl.prj_t[prj_ix].x += t.x*apix;
                ptcl.prj_t[prj_ix].y += t.y*apix;
            }
            else {
                ptcl.prj_t[prj_ix].x = t.x*apix;
                ptcl.prj_t[prj_ix].y = t.y*apix;
            }
        }
    }

    void set_classification(AliBuffer*ptr) {
        if( ptr->class_ix+1 == ptr->ptcl.n_refs ) {
            float max_cc = ptr->ptcl.ali_cc[0];
            ptr->ptcl.ref_cix() = 0;
            for(int i=0;i<ptr->ptcl.n_refs;i++) {
                if( max_cc < ptr->ptcl.ali_cc[i] ) {
                    max_cc = ptr->ptcl.ali_cc[i];
                    ptr->ptcl.ref_cix() = i;
                }
            }
        }
    }

};

class AliRdrWorker : public Worker {

public:
    ArgsAli::Info   *p_info;
    float           *p_stack;
    ParticlesSubset *p_ptcls;
    Tomogram        *p_tomo;
    RefMap          *p_refs;
    int gpu_ix;
    int max_K;
    int N;
    int M;
    int R;
    int P;
    int pad_type;
    int NP;
    int MP;

    bool drift2D;
    bool drift3D;

    float bp_pad;

    SubstackCrop    ss_cropper;

    AliRdrWorker() {
    }

    ~AliRdrWorker() {
    }

    void setup_global_data(int id,RefMap*in_p_refs,int in_R,int in_max_K,ArgsAli::Info*info,WorkerCommand*in_worker_cmd) {
        worker_id  = id;
        worker_cmd = in_worker_cmd;

        p_info   = info; 
        // info->n_threads = number_of_gpus * threads_per_gpu, in order to get a gpu index from a worker index we have to make an integer division by threads_per_gpu instead of remainder of the devision by info->n_threads, which will not change id, since id < info->n_threads all the time.
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

        drift2D = true;
        drift3D = true;

        if( info->type == 2 && !info->drift ) drift2D = false;
        if( info->type == 3 && !info->drift ) drift3D = false;
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
        AliBuffer buffer_a(N,max_K);
        AliBuffer buffer_b(N,max_K);
        PBarrier local_barrier(2);
        DoubleBufferHandler stack_buffer((void*)&buffer_a,(void*)&buffer_b,&local_barrier);

        AliGpuWorker gpu_worker;
        init_processing_worker(gpu_worker,&stack_buffer);

        int current_cmd;

        while( (current_cmd = worker_cmd->read_command()) >= 0 ) {
            switch(current_cmd) {
                case ALI_3D:
                    if(p_info->ignore_ref)
                        crop_loop_ignore_ref(stack_buffer,stream);
                    else
                        crop_loop(stack_buffer,stream);
                    break;
                case ALI_2D:
                    crop_loop(stack_buffer,stream);
                    break;
                default:
                    break;
            }
        }
        gpu_worker.wait();
    }

    void init_processing_worker(AliGpuWorker&gpu_worker,DoubleBufferHandler*stack_buffer) {
        bp_pad = p_info->fpix_roll/2;
        float bp_scale = ((float)NP)/((float)N);
        gpu_worker.worker_id  = worker_id;
        gpu_worker.worker_cmd = worker_cmd;
        gpu_worker.gpu_ix     = gpu_ix;
        gpu_worker.p_buffer   = stack_buffer;
        gpu_worker.N          = N;
        gpu_worker.M          = M;
        gpu_worker.P          = P;
        gpu_worker.R          = R;
        gpu_worker.p_refs     = p_refs;
        gpu_worker.pad_type   = pad_type;
        gpu_worker.ali_halves = p_info->ali_halves;
        gpu_worker.cc_stats   = p_info->cc_stats;
        gpu_worker.cc_type    = p_info->cc_type;
        gpu_worker.ctf_type   = p_info->ctf_type;
        gpu_worker.max_K      = max_K;
        gpu_worker.bandpass.x = fmax(bp_scale*p_info->fpix_min-bp_pad,0.0);
        gpu_worker.bandpass.y = fmin(bp_scale*p_info->fpix_max+bp_pad,((float)NP)/2);
        gpu_worker.bandpass.z = sqrt(p_info->fpix_roll);
        gpu_worker.ssnr.x     = p_info->ssnr_F*bp_scale;
        gpu_worker.ssnr.y     = p_info->ssnr_S;
        gpu_worker.drift2D    = drift2D;
        gpu_worker.drift3D    = drift3D;
        gpu_worker.cone.x     = p_info->cone_range;
        gpu_worker.cone.y     = p_info->cone_step;
        gpu_worker.inplane.x  = p_info->inplane_range;
        gpu_worker.inplane.y  = p_info->inplane_step;
        gpu_worker.ref_factor = p_info->refine_factor;
        gpu_worker.ref_level  = p_info->refine_level;
        gpu_worker.off_type   = p_info->off_type;
        gpu_worker.off_space  = p_info->off_space;
        gpu_worker.off_par.x  = p_info->off_x;
        gpu_worker.off_par.y  = p_info->off_y;
        gpu_worker.off_par.z  = p_info->off_z;
        gpu_worker.off_par.w  = p_info->off_s;
        gpu_worker.psym       = p_info->pseudo_sym;
        gpu_worker.tm_type    = p_info->tm_type;
        gpu_worker.tm_prefix  = p_info->tm_pfx;
        gpu_worker.tm_dim     = p_info->type;
        gpu_worker.tm_sigma   = p_info->tm_sigma;
        gpu_worker.dilate     = p_info->dilate;
        gpu_worker.start();
    }

    void crop_loop(DoubleBufferHandler&stack_buffer,GPU::Stream&stream) {
        stack_buffer.WO_sync(EMPTY);
        for(int i=worker_id;i<p_ptcls->n_ptcl;i+=p_info->n_threads) {
            for(int r=0;r<R;r++) {
                AliBuffer*ptr = (AliBuffer*)stack_buffer.WO_get_buffer();
                p_ptcls->get(ptr->ptcl,i);
                read_defocus(ptr);
                crop_substack(ptr,r);
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
            AliBuffer*ptr = (AliBuffer*)stack_buffer.WO_get_buffer();
            p_ptcls->get(ptr->ptcl,i);
            read_defocus(ptr);
            crop_substack(ptr);
            if( check_substack(ptr) ) {
                upload(ptr,stream.strm);
                stream.sync();
                stack_buffer.WO_sync(READY);
            }
            work_progress++;
            work_accumul++;
        }
        stack_buffer.WO_sync(DONE);
    }

    void read_defocus(AliBuffer*ptr) {
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

    void crop_substack(AliBuffer*ptr,const int ref_cix=-1) {
        V3f pt_tomo,pt_crop;
        M33f R_2D,R_base,R_gpu;

        ptr->class_ix = (ref_cix<0)?
                            ptr->ptcl.ref_cix(): // True
                            ref_cix;             // False

        ptr->r_ix = (p_info->ali_halves)?
                        2*ptr->class_ix + (ptr->ptcl.half_id()-1): // True
                        ptr->class_ix;                             // False

        pt_tomo = get_tomo_position(ptr->ptcl.pos(),ptr->ptcl.ali_t[ptr->class_ix],drift3D);
        pt_tomo = pt_tomo - p_tomo->tomo_position;
        ptr->set_tomo_pos(pt_tomo,p_tomo->tomo_center,p_tomo->pix_size);

        for(int k=0;k<ptr->K;k++) {
            if( ptr->ptcl.prj_w[k] > 0 ) {

                Math::eZYZ_Rmat(R_2D,ptr->ptcl.prj_eu[k]);
                R_base = R_2D * p_tomo->R[k];

                /*if( (ptr->ptcl.ptcl_id() == 44) && (k==20) ) {
                    printf("\nR_2D = np.zeros((3,3))\n");
                    printf("R_2D[0,:] = (%f,%f,%f)\n",R_2D(0,0),R_2D(0,1),R_2D(0,2));
                    printf("R_2D[1,:] = (%f,%f,%f)\n",R_2D(1,0),R_2D(1,1),R_2D(1,2));
                    printf("R_2D[2,:] = (%f,%f,%f)\n",R_2D(2,0),R_2D(2,1),R_2D(2,2));
                    printf("\nRtlt = np.zeros((3,3))\n");
                    printf("Rtlt[0,:] = (%f,%f,%f)\n",p_tomo->R[k](0,0),p_tomo->R[k](0,1),p_tomo->R[k](0,2));
                    printf("Rtlt[1,:] = (%f,%f,%f)\n",p_tomo->R[k](1,0),p_tomo->R[k](1,1),p_tomo->R[k](1,2));
                    printf("Rtlt[2,:] = (%f,%f,%f)\n",p_tomo->R[k](2,0),p_tomo->R[k](2,1),p_tomo->R[k](2,2));
                    printf("\nR_base = np.zeros((3,3))\n");
                    printf("R_base[0,:] = (%f,%f,%f)\n",R_base(0,0),R_base(0,1),R_base(0,2));
                    printf("R_base[1,:] = (%f,%f,%f)\n",R_base(1,0),R_base(1,1),R_base(1,2));
                    printf("R_base[2,:] = (%f,%f,%f)\n",R_base(2,0),R_base(2,1),R_base(2,2));
                }*/

                pt_crop = project_tomo_position(pt_tomo,p_tomo->R[k],p_tomo->t[k],ptr->ptcl.prj_t[k],drift2D);
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

        ptr->crowther_limit = 1/tanf( p_tomo->get_angle_step_rad() );
    }

    bool check_substack(AliBuffer*ptr) {
        bool rslt = false;
        for(int k=0;k<ptr->K;k++) {
            if( ptr->c_ali.ptr[k].w > 0 )
                rslt = true;
        }
        return rslt;
    }

    static V3f get_tomo_position(const Vec3&pos_base,const Vec3&shift,bool drift) {
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
                                     bool drift)
    {
        V3f pos_stack = R_tomo * pos_tomo + shift_tomo;
        if(drift) {
            pos_stack(0) += shift_2D.x;
            pos_stack(1) += shift_2D.y;
        }
        return pos_stack;
    }

    void upload(AliBuffer*ptr,cudaStream_t&strm) {
        GPU::upload_async(ptr->g_stk.ptr,ptr->c_stk.ptr,N*N*max_K,strm);
        GPU::upload_async(ptr->g_pad.ptr,ptr->c_pad.ptr,max_K    ,strm);
        GPU::upload_async(ptr->g_ali.ptr,ptr->c_ali.ptr,max_K    ,strm);
        GPU::upload_async(ptr->g_def.ptr,ptr->c_def.ptr,max_K    ,strm);
    }

};

class AliPool : public PoolCoordinator {

public:
    AliRdrWorker  *workers;
    ArgsAli::Info *p_info;
    WorkerCommand w_cmd;
    RefMap        *p_refs;
    int max_K;
    int N;
    int M;
    int R;
    int P;
    int n_ptcls;
    int NP;
    int MP;

    ProgressReporter progress;

    AliPool(ArgsAli::Info*info,References*in_p_refs,int in_max_K,int num_ptcls,StackReader&stkrdr,int in_num_threads)
     : PoolCoordinator(stkrdr,in_num_threads),
       w_cmd(2*in_num_threads+1),
       progress("    Aligning particles",num_ptcls)
    {
        workers  = new AliRdrWorker[in_num_threads];
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

    ~AliPool() {
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
        if( p_info->type == 3 )
            w_cmd.send_command(AliCmd::ALI_3D);
        if( p_info->type == 2 )
            w_cmd.send_command(AliCmd::ALI_2D);

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

#endif /// ALIGNER_H


