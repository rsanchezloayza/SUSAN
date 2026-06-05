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

#ifndef CTF_LINEARIZER_H
#define CTF_LINEARIZER_H

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
#include "math_cpu.h"
#include "estimate_ctf_args.h"
#include <algorithm>
#include <iostream>

#include "Eigen/Dense"

class CtfLinearizer1D{

public:
    GPU::GArrSingle  g_input;
    GPU::GArrSingle  g_linear;
    GPU::GArrSingle  g_weight;
    GPU::GArrSingle  g_bufferA;
    GPU::GArrSingle  g_bufferB;
    GPU::GArrSingle2 g_fourier;
    GPU::GArrSingle  g_power;

    GpuFFT::FFT1D rfft;

    float  *c_linear;
    float  *c_defocus;

    float N;
    float M;
    float K;

    float apix;
    float new_apix;
    float max_nyquist;
    float lin_fpix_to_defocus;

    int   verbose;

    char filename[SUSAN_FILENAME_LENGTH];

    float u0;
    float s0;
    int   k0;

    CtfLinearizer1D(int gpu_ix, int n, int k, float w_target_apix=3.5f) {

        N = n;
        M = n/2 + 1;
        K = k;

        apix = 1.0;
        new_apix = w_target_apix;

        GPU::set_device(gpu_ix);

        rfft.alloc(N,K);

        g_power.alloc(M*K);
        g_input.alloc(M*N*K);
        g_linear.alloc(N*K);
        g_weight.alloc(N*K);
        g_bufferA.alloc(M*N*K);
        g_bufferB.alloc(M*N*K);
        g_fourier.alloc(M*K);

        c_linear  = new float[int(N*K)];
        c_defocus = new float[int(K)];
    }

    ~CtfLinearizer1D() {
        delete [] c_linear;
        delete [] c_defocus;
    }

    void load_info(ArgsCTF::Info*info,Tomogram*p_tomo) {

        float lambda = Math::get_lambda(p_tomo->KV);

        verbose   = info->verbose;

        apix = p_tomo->pix_size;
        max_nyquist = apix/new_apix;
        lin_fpix_to_defocus = 2.0f*new_apix*new_apix/lambda;
    }

    float get_rough_estimate(const char*out_dir,float*input) {

        preprocess(input);
        radial_average();
        calculate_power_spectrum();

        cudaMemcpy( (void*)c_linear, (const void*)g_power.ptr, sizeof(float)*M*K, cudaMemcpyDeviceToHost);
        save_gpu_mrc(c_linear,g_power.ptr,M,K,1,out_dir,"ps_lin_raw.mrc",2);

        float max_val=0;
        float *tmp = new float[int(M)];
        int ix0,ix1;

        /// AVERAGE ALL PROJECTIONS
        for(int m=0;m<M;m++)
            tmp[m] = 0;

        for(int k=0;k<K;k++)
            for(int m=0;m<M;m++) {
                if( c_linear[m+k*int(M)] > max_val ) {
                    max_val = c_linear[m+k*int(M)];
                    k0 = k;
                }
                tmp[m] += c_linear[m+k*int(M)];
            }

        max_val = 0;
        for(int m=0;m<M;m++) {
            if( tmp[m] > max_val ) {
                max_val = tmp[m];
                u0 = m;
            }
        }

        /// GET INITIAL U0 AND S0
        estimate_mu_sigma(u0,s0,tmp,fmax(int(u0)-4,0),fmin(int(u0)+4,M-1));

        /// GO FROM CENTRAL TO THE END
        float u = u0;
        float s = s0;
        for(int k=k0;k<K;k++) {
            int P = int(ceilf(3*s));
            ix0 = int(fmax(fmin(u0,u)-P,0));
            ix1 = int(fmin(fmax(u0,u)+P,M-1));
            estimate_mu_sigma(u,s,c_linear+(k*int(M)),ix0,ix1);
            c_defocus[k] = lin_fpix_to_defocus*u;
        }

        /// GO FROM CENTRAL TO THE BEGINNING
        u = u0;
        s = s0;
        for(int k=k0-1;k>=0;k--) {
            int P = int(ceilf(3*s));
            ix0 = int(fmax(fmin(u0,u)-P,0));
            ix1 = int(fmin(fmax(u0,u)+P,M-1));
            estimate_mu_sigma(u,s,c_linear+(k*int(M)),ix0,ix1);
            c_defocus[k] = lin_fpix_to_defocus*u;
        }

        delete [] tmp;
        return lin_fpix_to_defocus*u0;
    }

protected:
    void preprocess(float*input) {
        int3 siz = make_int3(M,N,K);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,siz.x,siz.y,siz.z);
        cudaMemcpy( (void*)g_input.ptr, (const void*)(input), sizeof(float)*M*N*K, cudaMemcpyHostToDevice);
        float num = 0.0625f/2;
        float scl = 10.02548736875733f;
        GpuKernels::conv_gauss_1D_Y<<<grd,blk>>>(g_bufferA.ptr,g_input.ptr,num,scl,siz);
        GpuKernels::conv_gauss_1D_X<<<grd,blk>>>(g_bufferB.ptr,g_bufferA.ptr,num,scl,siz,true);
        GpuKernels::substract<<<grd,blk>>>(g_input.ptr,g_bufferB.ptr,siz);
    }

    void radial_average() {
        g_linear.clear();
        g_weight.clear();

        int3 siz_ps = make_int3(M,N,K);
        dim3 blk_ps = GPU::get_block_size_2D();
        dim3 grd_ps = GPU::calc_grid_size(blk_ps,siz_ps.x,siz_ps.y,siz_ps.z);

        GpuKernels::radial_ps_avg_linearized<<<grd_ps,blk_ps>>>(g_linear.ptr,g_weight.ptr,g_input.ptr,max_nyquist,siz_ps);

        int3 siz_rad = make_int3(N,K,1);
        dim3 blk_rad = GPU::get_block_size_2D();
        dim3 grd_rad = GPU::calc_grid_size(blk_rad,siz_rad.x,siz_rad.y,siz_rad.z);

        GpuKernels::divide<<<grd_rad,blk_rad>>>(g_linear.ptr,g_weight.ptr,siz_rad);
        GpuKernels::arr_hanning<<<grd_rad,blk_rad>>>(g_linear.ptr,siz_rad);
    }

    void calculate_power_spectrum() {
        int3 siz_rad = make_int3(M,K,1);
        dim3 blk_rad = GPU::get_block_size_2D();
        dim3 grd_rad = GPU::calc_grid_size(blk_rad,siz_rad.x,siz_rad.y,siz_rad.z);

        rfft.exec(g_fourier.ptr,g_linear.ptr);
        GpuKernels::divide<<<grd_rad,blk_rad>>>(g_fourier.ptr,N,siz_rad);
        GpuKernels::load_ps<<<grd_rad,blk_rad>>>(g_power.ptr,g_fourier.ptr,siz_rad);
    }

    void estimate_mu_sigma(float&mu,float&sigma,float*p_data,int ix0,int ix1) {
        float acc = 0;
        mu    = 0;
        sigma = 0;

        for(int ix=ix0;ix<=ix1;ix++) {
            mu  += p_data[ix]*ix;
            acc += p_data[ix];
        }
        mu = mu/acc;

        for(int ix=ix0;ix<=ix1;ix++) {
            sigma += p_data[ix]*((ix-mu)*(ix-mu)) ;
        }
        sigma = sigma/acc;

        sigma = sqrtf(sigma);
    }

    void save_gpu_mrc(single*p_cpu,const single*p_gpu,const int m,const int n,const int k,const char*out_dir,const char*name,const int req_verb) {
        cudaMemcpy((void*)p_cpu, (const void*)p_gpu, sizeof(float)*m*n*k, cudaMemcpyDeviceToHost);
        if( verbose >= req_verb ) {
            sprintf(filename,"%s/%s",out_dir,name);
            Mrc::write(p_cpu,m,n,k,filename);
        }
    }


};

class CtfLinearizer{

protected:
    class CtfRefiner {
    public:
        int M;
        int N;
        int K;
        float*buf_ctf;
        float*buf_metric;
        float*buf_m_var;
        float*buf_cum_s1;
        float*buf_cum_s2;

        float apix;
        float lambda_pi;
        float lambda3_Cs_pi_2;
        float AC;
        float CA;

        int last_th;

        CtfRefiner(int m, int n, int k, float apix_in, float lambda_kv, float ac, float cs) {
            M = m;
            N = n;
            K = k;
            buf_ctf    = new float[M];
            buf_metric = new float[M];
            buf_m_var  = new float[M];
            buf_cum_s1 = new float[M];
            buf_cum_s2 = new float[M];

            AC = ac;
            CA = sqrt(1.0-AC*AC);
            apix = apix_in;
            lambda_pi = lambda_kv*M_PI;
            lambda3_Cs_pi_2 = lambda_kv*lambda_kv*lambda_kv*(cs*1e7)*M_PI/2;
        }

        ~CtfRefiner() {
            delete [] buf_ctf;
            delete [] buf_metric;
            delete [] buf_m_var;
            delete [] buf_cum_s1;
            delete [] buf_cum_s2;
        }

        void refine_ctf_hybrid(float&best_score,float&delta_def,float&phase_shift,int&max_fpix,const float*p_data,float init_defocus,bool est_phase=true) {

            const float def_steps[3]   = {200.0f, 50.0f, 10.0f};
            const float phase_steps[3] = {2.0f, 0.5f, 0.5f}; // degrees

            delta_def   = 0.0f;
            phase_shift = 0.0f;

            best_score = compute_score(delta_def,phase_shift,p_data,init_defocus);
            max_fpix   = last_th;

            float phase_span = (float)M_PI/2.0f;

            for(int level=0;level<3;level++) {

                float d_step = def_steps[level];
                float p_step = phase_steps[level]*DEG2RAD;

                delta_def  = parabolic_defocus(delta_def,phase_shift,d_step,p_data,init_defocus);
                best_score = compute_score(delta_def,phase_shift,p_data,init_defocus);
                max_fpix   = last_th;

                if( est_phase ) {
                    float local_best_score = best_score;
                    float best_phase_local = phase_shift;

                    int n_phase = (int)ceilf(phase_span/p_step);

                    for (int i=-n_phase;i<=n_phase;i++) {

                        float phase_i = phase_shift + p_step*i;
                        float score_i = compute_score(delta_def, phase_i,p_data,init_defocus);

                        if( score_i>local_best_score) {
                            local_best_score = score_i;
                            best_phase_local = phase_i;
                        }
                    }

                    phase_shift = best_phase_local;
                    best_score  = local_best_score;

                    phase_span = p_step * 4.0f;
                }
            }

            best_score = compute_score(delta_def,phase_shift,p_data,init_defocus);
            max_fpix   = last_th;
        }

    protected:
        void ctf_1D(float def_angs,float phase_shift_rad) {

            float df = 1.0f/(N*apix);
            float lambda_pi_def = lambda_pi*def_angs;

            for (int i=0;i<M;i++) {
                float s  = i*df;
                float s2 = s*s;
                float s4 = s2*s2;

                float gamma = lambda_pi_def*s2 - lambda3_Cs_pi_2*s4 + phase_shift_rad;
                buf_ctf[i] = -(CA*sinf(gamma)+AC*cosf(gamma));
            }
        }

        void local_std(int window) {
            int wsize = 2*window+1;

            for(int i=0;i<M;i++) {

                float s = 0.0f;
                float s2 = 0.0f;

                for(int k=-window;k<=window;k++) {

                    int j = i+k;

                    if (j < 0)
                        j = -j-1;
                    else if (j>=M)
                        j = 2*M-j-1;

                    float val = buf_metric[j];
                    s  += val;
                    s2 += val*val;
                }

                float mean = s/wsize;
                float var  = s2/wsize - mean*mean;

                if (var < 0.0f) var = 0.0f;

                buf_m_var[i] = sqrtf(var);
            }
        }

        int var_split_index() {

            if(M<4)
                return M;

            float total  = 0.0f;
            float total2 = 0.0f;

            for(int i=0;i<M;i++) {
                float v = buf_m_var[i];
                total  += v;
                total2 += v*v;
                buf_cum_s1[i]  = total;
                buf_cum_s2[i] = total2;
            }

            int best_k = 1;
            float best_score = 99999.9f;

            for(int k=2;k<=M-2;k++) {

                // Left segment
                int n1 = k;
                float sum1  = buf_cum_s1 [k-1];
                float sum12 = buf_cum_s2[k-1];
                float mean1 = sum1 /n1;
                float var1  = sum12/n1 - mean1*mean1;
                if (var1 < 0.0f) var1 = 0.0f;

                // Right segment
                int n2 = M - k;
                float sum2  = total -sum1;
                float sum22 = total2-sum12;
                float mean2 = sum2 /n2;
                float var2  = sum22/n2 - mean2*mean2;
                if (var2 < 0.0f) var2 = 0.0f;

                float score = var1 + var2;

                if (score < best_score) {
                    best_score = score;
                    best_k = k;
                }
            }

            return best_k;
        }

        float var_split_improvement(int best_k) {

            if (best_k <= 0 || best_k >= M)
                return 0.0f;

            float total_sum    = 0.0f;
            float total_sq_sum = 0.0f;

            for(int i=0;i<M;i++) {
                float val     = buf_m_var[i];
                total_sum    += val;
                total_sq_sum += val*val;
            }

            float mean_global = total_sum/M;
            float var_global  = total_sq_sum/M - mean_global * mean_global;
            float rss1 = M*var_global;

            if (rss1 <= 0.0f)
                return 0.0f;

            float left_sum    = 0.0f;
            float left_sq_sum = 0.0f;
            for (int i=0;i<best_k;i++) {
                float val    = buf_m_var[i];
                left_sum    += val;
                left_sq_sum += val*val;
            }

            float mean1 = left_sum/best_k;
            float var1  = left_sq_sum/best_k - mean1*mean1;

            int   right_n      = M-best_k;
            float right_sum    = total_sum-left_sum;
            float right_sq_sum = total_sq_sum-left_sq_sum;

            float mean2 = right_sum/right_n;
            float var2  = right_sq_sum/right_n - mean2*mean2;

            if(var1<0.0f) var1=0.0f;
            if(var2<0.0f) var2=0.0f;

            float rss2 = best_k*var1 + right_n*var2;

            return (rss1-rss2)/rss1;
        }

        float compute_score(float delta_def,float phase,const float*p_data,float init_defocus) {

            ctf_1D(init_defocus+delta_def,phase);

            for(int i=0;i<M;i++) {
                float c = buf_ctf[i];
                buf_metric[i] = 1.0f - fabsf(p_data[i] - (c*c));
            }

            local_std(3);

            last_th = var_split_index();

            if (var_split_improvement(last_th) < 0.05f)
                last_th = M;

            float score = 0.0f;
            for(int i=0;i<last_th; i++)
                score += buf_metric[i];

            return score/float(M);
        }

        float parabolic_defocus(float delta_def,float phase,float def_step,const float* p_data,float init_defocus) {

            float x = delta_def;

            for(int iter=0;iter<3; iter++) {
                float f_0 = compute_score(x         ,phase,p_data,init_defocus);
                float f_p = compute_score(x+def_step,phase,p_data,init_defocus);
                float f_m = compute_score(x-def_step,phase,p_data,init_defocus);

                float den = (f_m-2.0f*f_0+f_p);

                if (fabsf(den) < 1e-12f)
                    break;

                float dx = 0.5f*def_step*(f_m-f_p)/den;

                if (dx >  def_step) dx =  def_step;
                if (dx < -def_step) dx = -def_step;

                x += dx;
            }

            return x;
        }

    };

public:
    float N;
    float M;
    float K;
    float R;
    float ang_step;

    float apix;
    float new_apix;
    float max_nyquist;
    float lin_fpix_to_defocus;

    int def_ix_min;
    int def_ix_max;

    int  verbose;
    bool est_phase_shift;

    float   *c_ini_idx;
    Defocus *c_estimate;

    float lambda_kv;
    float lambda_pi;
    float lambda3_Cs_pi_2;

    float CS;
    float AC;
    float CA;

    char filename[SUSAN_FILENAME_LENGTH];

    CtfLinearizer(int gpu_ix, int n, int k, float w_target_apix=3.5f, float r=90) {

        N = n;
        M = n/2 + 1;
        K = k;
        R = r;

        apix = 1.0;
        new_apix = w_target_apix;
        ang_step = M_PI/R;

        GPU::set_device(gpu_ix);

        c_ini_idx  = new float[int(K)];
        c_estimate = new Defocus[int(K)];
        memset((void*)c_estimate,0,sizeof(Defocus)*int(K));
    }

    ~CtfLinearizer() {

        delete [] c_ini_idx;
        delete [] c_estimate;
    }

    void load_info(ArgsCTF::Info*info,Tomogram*p_tomo) {

        float lambda = Math::get_lambda(p_tomo->KV);

        apix = p_tomo->pix_size;
        verbose = info->verbose;
        est_phase_shift = info->est_phase_shift;
        max_nyquist = apix/new_apix;
        lin_fpix_to_defocus = 2.0f*new_apix*new_apix/lambda;

        def_ix_min = int( floorf( info->def_min/lin_fpix_to_defocus ) );
        def_ix_max = int( ceilf ( info->def_max/lin_fpix_to_defocus ) );

        CS = p_tomo->CS;
        AC = p_tomo->AC;
        CA = sqrt(1.0-AC*AC);
        lambda_kv = Math::get_lambda(p_tomo->KV);
        lambda_pi = lambda_kv*M_PI;
        lambda3_Cs_pi_2 = lambda_kv*lambda_kv*lambda_kv*(CS*1e7)*M_PI/2;
    }

    void initial_estimation(const char*out_dir,float*input) {
        GPU::GArrSingle  g_input;
        GPU::GArrSingle  g_ps;

        GpuFFT::FFT1D rfft;

        float  *c_linear;

        rfft.alloc(N,K);

        g_ps.alloc(M*K);
        g_input.alloc(M*N*K);

        c_linear  = new float[int(N*K)];

        /// LOAD AND REMOVE BACKGROUND
        load_remove_bg_normalize(g_input,input);

        /// RADIAL AVERAGE AND LINEARIZE THEN GET POWER SPECTRUM
        radial_avg_and_power_spectrum(g_ps,g_input);

        /// DOWNLOAD TO CPU AND FIND INITIAL PEAKS
        cudaMemcpy( (void*)c_linear, (const void*)g_ps.ptr, sizeof(float)*M*K, cudaMemcpyDeviceToHost);
        if( verbose > 1 ) {
            sprintf(filename,"%s/ps_lin_norm.mrc",out_dir);
            Mrc::write(c_linear,M,K,1,filename);
            Mrc::set_apix(filename,new_apix,M,K,1);
        }
        track_peaks_in_linear_ps(c_linear);

        delete [] c_linear;
    }

    void process(const char*out_dir,float*input,Tomogram*p_tomo,int k0,bool is_overfocus) {

        GPU::GArrSingle  g_input;
        GPU::GArrSingle  g_linear;
        GPU::GArrSingle2 g_polar_as;
        GPU::GArrSingle2 g_analytical;
        GPU::GArrSingle  g_ps;

        GpuFFT::FFT1D  rfft;
        rfft.alloc(N,R*K);

        g_input.alloc(M*N*K);
        g_linear.alloc(N*N*K);
        g_analytical.alloc(N*N*K);
        g_polar_as.alloc(N*R*K);
        g_ps.alloc(N*N*K);

        float*c_buffer = new float[int(fmax(2*N*N*K,N*R*K))];

        /// LOAD AND REMOVE BACKGROUND
        load_remove_bg_normalize(g_input,input);
        save_gpu_mrc(c_buffer,g_input.ptr,M,N,K,out_dir,"ctf_normalized.mrc",1);

        /// LINEARIZE IN 2D
        fully_linearize(g_linear,g_input);
        save_gpu_mrc(c_buffer,g_linear.ptr,N,N,K,out_dir,"ctf_linearized.mrc",1);

        /// CONVERT TO POLAR
        to_polar_complex(g_polar_as,g_linear);
        save_gpu_mrc(c_buffer,g_polar_as.ptr,N,R,K,out_dir,"ctf_linear_polar.mrc",2);

        /// GET THE ANALYTICAL SIGNAL AND THE ELLIPSOIDAL POWER SPECTRUM
        polar_to_analytical_signal_to_cartesian(g_analytical,g_polar_as);
        get_power_spectrum(g_ps,g_analytical);
        save_gpu_mrc(c_buffer,g_ps.ptr,N,N,K,out_dir,"ctf_ps.mrc",2);

        /// DOWNLOAD FOR ELLIPSOIDAL FITTING
        cudaMemcpy( (void*)c_buffer, (const void*)g_ps.ptr, sizeof(float)*N*N*K, cudaMemcpyDeviceToHost);
        fit_ellipsoid(c_buffer,out_dir,is_overfocus);

        /// REFINE AND ESTIMATE PHASE SHIFT
        estimate_phase_shift(c_buffer,g_input,out_dir);

        /// SAVE RESULTS
        save_fitting(out_dir,c_buffer,g_input);
        save_defocus_result(out_dir);

        delete [] c_buffer;
    }


protected:
    void save_cpu_mrc(single*p_cpu,const int m,const int n,const int k,const char*out_dir,const char*name,const int req_verb) {
        if( verbose >= req_verb ) {
            sprintf(filename,"%s/%s",out_dir,name);
            Mrc::write(p_cpu,m,n,k,filename);
        }
    }

    void save_gpu_mrc(single*p_cpu,const single*p_gpu,const int m,const int n,const int k,const char*out_dir,const char*name,const int req_verb) {
        cudaMemcpy((void*)p_cpu, (const void*)p_gpu, sizeof(float)*m*n*k, cudaMemcpyDeviceToHost);
        save_cpu_mrc(p_cpu,m,n,k,out_dir,name,req_verb);
    }

    void save_gpu_mrc(single*p_cpu,const float2*p_gpu,const int m,const int n,const int k,const char*out_dir,const char*name,const int req_verb,bool save_real=true) {
        cudaMemcpy((void*)p_cpu, (const void*)p_gpu, sizeof(float2)*m*n*k, cudaMemcpyDeviceToHost);
        if(save_real)
            for(int i=0;i<(m*n*k);i++)
                p_cpu[i] = p_cpu[2*i];
        else
            for(int i=0;i<(m*n*k);i++)
                p_cpu[i] = p_cpu[(2*i)+1];
        save_cpu_mrc(p_cpu,m,n,k,out_dir,name,req_verb);
    }

    void estimate_mu_sigma_1D(float&mu,float&sigma,float*p_data,int ix0,int ix1) {
        float acc = 0;
        mu    = 0;
        sigma = 0;

        for(int ix=ix0;ix<=ix1;ix++) {
            mu  += p_data[ix]*ix;
            acc += p_data[ix];
        }
        mu = mu/acc;

        for(int ix=ix0;ix<=ix1;ix++) {
            sigma += p_data[ix]*((ix-mu)*(ix-mu)) ;
        }
        sigma = sigma/acc;

        sigma = sqrtf(sigma);
    }

    void track_peaks_in_linear_ps(float*c_linear) {
        float max_val=0;
        float *p_avg = new float[int(M)];
        float u0, s0;
        int ix0,ix1;
        int k0;

        /// AVERAGE ALL PROJECTIONS
        for(int m=0;m<M;m++)
            p_avg[m] = 0;

        for(int k=0;k<K;k++)
            for(int m=0;m<M;m++) {
                if( c_linear[m+k*int(M)] > max_val ) {
                    max_val = c_linear[m+k*int(M)];
                    k0 = k;
                }
                p_avg[m] += c_linear[m+k*int(M)];
            }

        max_val = 0;
        for(int m=0;m<M;m++) {
            if( p_avg[m] > max_val ) {
                max_val = p_avg[m];
                u0 = m;
            }
        }

        /// GET INITIAL U0 AND S0
        estimate_mu_sigma_1D(u0,s0,p_avg,fmax(int(u0)-4,0),fmin(int(u0)+4,M-1));

        /// GO FROM CENTRAL TO THE END
        float u = u0;
        float s = s0;
        for(int k=k0;k<K;k++) {
            int P = int(ceilf(3*s));
            ix0 = int(fmax(fmin(u0,u)-P,0));
            ix1 = int(fmin(fmax(u0,u)+P,M-1));
            estimate_mu_sigma_1D(u,s,c_linear+(k*int(M)),ix0,ix1);
            c_ini_idx[k] = u;
        }

        /// GO FROM CENTRAL TO THE BEGINNING
        u = u0;
        s = s0;
        for(int k=k0-1;k>=0;k--) {
            int P = int(ceilf(3*s));
            ix0 = int(fmax(fmin(u0,u)-P,0));
            ix1 = int(fmin(fmax(u0,u)+P,M-1));
            estimate_mu_sigma_1D(u,s,c_linear+(k*int(M)),ix0,ix1);
            c_ini_idx[k] = u;
        }

        delete [] p_avg;
    }

    void radial_avg_and_power_spectrum(GPU::GArrSingle&g_ps,GPU::GArrSingle&g_input) {
        GPU::GArrSingle  g_acc;
        GPU::GArrSingle  g_wgt;
        GPU::GArrSingle2 g_fou;
        g_acc.alloc(N*K);
        g_wgt.alloc(N*K);
        g_fou.alloc(M*K);

        GpuFFT::FFT1D rfft;
        rfft.alloc(N,K);

        g_acc.clear();
        g_wgt.clear();

        dim3 blk = GPU::get_block_size_2D();

        int3 siz_ps = make_int3(M,N,K);
        dim3 grd_ps = GPU::calc_grid_size(blk,siz_ps);

        GpuKernels::radial_ps_avg_linearized<<<grd_ps,blk>>>(g_acc.ptr,g_wgt.ptr,g_input.ptr,max_nyquist,siz_ps);

        int3 siz_rad = make_int3(N,K,1);
        dim3 grd_rad = GPU::calc_grid_size(blk,siz_rad);

        GpuKernels::divide<<<grd_rad,blk>>>(g_acc.ptr,g_wgt.ptr,siz_rad);
        GpuKernels::arr_hanning<<<grd_rad,blk>>>(g_acc.ptr,siz_rad);

        /// CALCULATE POWER SPECTRUM AND DOWNLOAD TO CPU
        int3 siz_lin = make_int3(M,K,1);
        dim3 grd_lin = GPU::calc_grid_size(blk,siz_lin);

        rfft.exec(g_fou.ptr,g_acc.ptr);
        GpuKernels::divide<<<grd_lin,blk>>>(g_fou.ptr,N,siz_lin);
        GpuKernels::load_ps<<<grd_lin,blk>>>(g_ps.ptr,g_fou.ptr,siz_lin);
    }

    void load_remove_bg_normalize(GPU::GArrSingle&rslt,float*input) {

        GPU::GArrSingle g_bufferA;
        GPU::GArrSingle g_bufferB;
        g_bufferA.alloc(M*N*K);
        g_bufferB.alloc(M*N*K);

        cudaMemcpy( (void*)rslt.ptr, (const void*)(input), sizeof(float)*M*N*K, cudaMemcpyHostToDevice);

        int3 siz = make_int3(M,N,K);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,siz.x,siz.y,siz.z);
        float num = 0.0625f/2;
        float scl = 10.02548736875733f;

        GpuKernels::conv_gauss_1D_Y<<<grd,blk>>>(g_bufferA.ptr,rslt.ptr,num,scl,siz);
        GpuKernels::conv_gauss_1D_X<<<grd,blk>>>(g_bufferB.ptr,g_bufferA.ptr,num,scl,siz,true);
        GpuKernels::substract<<<grd,blk>>>(rslt.ptr,g_bufferB.ptr,siz);
        g_bufferA.clear();
        g_bufferB.clear();
        GpuKernels::get_avg_std<<<grd,blk>>>(g_bufferA.ptr,g_bufferB.ptr,rslt.ptr,siz);
        GpuKernels::zero_avg_one_std<<<grd,blk>>>(rslt.ptr,g_bufferA.ptr,g_bufferB.ptr,siz);
    }

    void fully_linearize(GPU::GArrSingle&rslt,GPU::GArrSingle&hermitian_input) {
        GPU::GArrSingle g_acc;
        GPU::GArrSingle g_wgt;
        GPU::GArrSingle g_tmp;

        g_acc.alloc(N*N*K);
        g_wgt.alloc(N*N*K);

        g_acc.clear();
        g_wgt.clear();

        dim3 blk = GPU::get_block_size_2D();

        int3 siz_hermitian = make_int3(M,N,K);
        dim3 grd_hermitian = GPU::calc_grid_size(blk,siz_hermitian);

        int3 siz_full = make_int3(N,N,K);
        dim3 grd_full = GPU::calc_grid_size(blk,siz_full);

        GpuKernelsCtf::ctf_linearize_ps<<<grd_hermitian,blk>>>(g_acc.ptr,g_wgt.ptr,hermitian_input.ptr,apix,new_apix,siz_hermitian);

        GpuKernels::divide<<<grd_full,blk>>>(g_acc.ptr,g_wgt.ptr,siz_full);
        GpuKernels::stk_hanning<<<grd_full,blk>>>(g_acc.ptr,siz_full);
        GpuKernels::stk_medfilt_k3<<<grd_full,blk>>>(rslt.ptr,g_acc.ptr,siz_full);
    }

    void to_polar_complex(GPU::GArrSingle2&polar_complex,GPU::GArrSingle&input) {
        dim3 blk = GPU::get_block_size_2D();
        int3 siz = make_int3(N,R,K);
        dim3 grd = GPU::calc_grid_size(blk,siz);
        int3 siz_input = make_int3(N,N,K);
        GpuKernels::to_polar_complex<<<grd,blk>>>(polar_complex.ptr,input.ptr,ang_step,siz_input,siz);
    }

    void polar_to_analytical_signal_to_cartesian(GPU::GArrSingle2&stk_complex,GPU::GArrSingle2&polar_complex) {
        GPU::GArrSingle2    g_fourier;
        GpuFFT::FFT1D_full  fft;
        GpuFFT::IFFT1D_full ifft;

        g_fourier.alloc(N*R*K);
        fft.alloc(N,R*K);
        ifft.alloc(N,R*K);

        dim3 blk = GPU::get_block_size_2D();

        int3 siz_polar = make_int3(N,R*K,1);
        dim3 grd_polar = GPU::calc_grid_size(blk,siz_polar);

        int3 siz_cart = make_int3(N,N,K);
        dim3 grd_cart = GPU::calc_grid_size(blk,siz_cart);

        fft.exec(g_fourier.ptr,polar_complex.ptr);
        GpuKernels::analytical_signal_1D<<<grd_polar,blk>>>(g_fourier.ptr,M,N,R,K);
        ifft.exec(polar_complex.ptr,g_fourier.ptr);
        GpuKernels::divide<<<grd_polar,blk>>>(polar_complex.ptr,N,siz_polar);
        GpuKernels::polar_to_cart_complex<<<grd_cart,blk>>>(stk_complex.ptr,polar_complex.ptr,ang_step,N,R,K);

    }

    void get_power_spectrum(GPU::GArrSingle&g_ps,GPU::GArrSingle2&stk_complex,int scale=2) {

        GPU::GArrSingle2   g_pad;
        GPU::GArrSingle2   g_fourier;
        GPU::GArrSingle    g_bufferA;
        GPU::GArrSingle    g_bufferB;
        GPU::GArrSingle    g_bufferC;
        GpuFFT::FFT2D_full fft2;

        dim3 blk = GPU::get_block_size_2D();

        int3 siz_raw = make_int3(N,N,K);
        dim3 grd_raw = GPU::calc_grid_size(blk,siz_raw);

        int3 siz_pad = make_int3(scale*N,scale*N,K);
        dim3 grd_pad = GPU::calc_grid_size(blk,siz_pad);

        g_pad.alloc(scale*scale*N*N*K);
        g_bufferA.alloc(scale*scale*N*N*K);
        g_bufferB.alloc(scale*scale*N*N*K);
        g_bufferC.alloc(scale*scale*N*N*K);
        g_fourier.alloc(scale*scale*N*N*K);
        fft2.alloc(scale*N,scale*N,K);

        int  padding = ((int)(scale*N-N))/2;
        int3 half_pad;
        half_pad.x = padding;
        half_pad.y = padding;
        half_pad.z = 0;

        g_pad.clear();
        GpuKernels::load_pad<<<grd_raw,blk>>>(g_pad.ptr,stk_complex.ptr,half_pad,siz_raw,siz_pad);

        fft2.exec(g_fourier.ptr,g_pad.ptr);

        GpuKernels::load_ps<<<grd_pad,blk>>>(g_bufferC.ptr,g_fourier.ptr,siz_pad);
        GpuKernels::divide<<<grd_pad,blk>>>(g_bufferC.ptr,scale*scale*N*N,siz_pad);
        GpuKernels::fftshift2D<<<grd_pad,blk>>>(g_bufferC.ptr,siz_pad);

        /*
        import numpy as np
        from scipy.ndimage import convolve

        cc = np.zeros((31,31))
        cc[15,15] = 1

        sigma = 4
        t   = np.arange(-15,16)
        num = 1/(2*(sigma**2))
        wx  = np.exp(-num*t*t)[:,np.newaxis]
        wy  = wx.T
        est = convolve(convolve(cc,wx),wy)
        print('num',num)
        print('scl',np.sqrt(est.sum()))
        */

        float num = 0.03125;
        float scl = 10.025487368757329f;

        GpuKernels::conv_gauss_1D_Y<<<grd_pad,blk>>>(g_bufferA.ptr,g_bufferC.ptr,num,scl,siz_pad);
        GpuKernels::conv_gauss_1D_X<<<grd_pad,blk>>>(g_bufferB.ptr,g_bufferA.ptr,num,scl,siz_pad,false);
        GpuKernels::substract<<<grd_pad,blk>>>(g_bufferC.ptr,g_bufferB.ptr,siz_pad);

        GpuKernels::remove_pad_stk<<<grd_raw,blk>>>(g_ps.ptr,g_bufferC.ptr,padding,siz_raw,siz_pad);

    }

    void init_estimate(int scale) {
        for(int k=0;k<K;k++){
            c_estimate[k].U = scale*c_ini_idx[k];
            c_estimate[k].V = scale*c_ini_idx[k];
            c_estimate[k].angle = 0.0;
        }
    }

    float rad_from_ellipsoid(int ix,int k) {
        float num = c_estimate[k].U * c_estimate[k].V;
        float ang = (ang_step*ix) - c_estimate[k].angle;
        float ca  = c_estimate[k].V*cosf(ang);
        float sa  = c_estimate[k].U*sinf(ang);
        float den = sqrtf( ca*ca + sa*sa );
        return num/den;
    }

    float sample_peak(int ix,int k,float*p_frame,float search_range=3) {
        int   n = (int)N;
        float rad = rad_from_ellipsoid(ix,k);
        float rad_min = rad - search_range;
        float rad_max = rad + search_range;

        float num=0;
        float den=0;
        float ca = cosf(ang_step*ix);
        float sa = sinf(ang_step*ix);

        for(float r=rad_min;r<=rad_max;r++) {
            float x = r*ca + N/2;
            float y = r*sa + N/2;

            int ix0 = (int)floorf(x);
            int iy0 = (int)floorf(y);

            if (ix0 < 0 || ix0 >= n-1 || iy0 < 0 || iy0 >= n - 1)
                continue;

            float wx = x - ix0;
            float wy = y - iy0;

            float v00 = p_frame[ ix0   + n*(iy0  ) ];
            float v10 = p_frame[ ix0+1 + n*(iy0  ) ];
            float v01 = p_frame[ ix0   + n*(iy0+1) ];
            float v11 = p_frame[ ix0+1 + n*(iy0+1) ];

            float val = 0.0f;

            val += v00*(1-wx)*(1-wy);
            val += v10*(  wx)*(1-wy);
            val += v01*(1-wx)*(  wy);
            val += v11*(  wx)*(  wy);

            if(val>0) {
                num += (r*val);
                den += val;
            }
        }

        if(den < SUSAN_FLOAT_TOL)
            return rad;
        else
            return num/den;
    }

    Eigen::Matrix2f get_covariance(float *r_arr) {

        float Cxx = 0.0f;
        float Cyy = 0.0f;
        float Cxy = 0.0f;

        for(int ix=0;ix<2*int(R);ix++) {

            float r = r_arr[ix];
            float a = ang_step*ix;

            float x = r*cosf(a);
            float y = r*sinf(a);

            Cxx += x*x;
            Cyy += y*y;
            Cxy += x*y;
        }

        float invN = 1.0f / (float)(2*R);

        Cxx *= invN;
        Cyy *= invN;
        Cxy *= invN;

        Eigen::Matrix2f C;
        C(0,0) = Cxx;
        C(0,1) = Cxy;
        C(1,0) = Cxy;
        C(1,1) = Cyy;

        return C;
    }

    void update_estimate_from_covariance(int k,Eigen::Matrix2f&cov) {

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(cov);

        Eigen::Vector2f eigvals = solver.eigenvalues();
        Eigen::Matrix2f eigvect = solver.eigenvectors();

        c_estimate[k].U = sqrtf(2.0f*eigvals(1));
        c_estimate[k].V = sqrtf(2.0f*eigvals(0));
        c_estimate[k].angle = atan2f(eigvect(1,1),eigvect(0,1));
    }

    void fit_ellipsoid(float*p_data,const char*out_dir,bool is_overfocus,int scale=2) {

        init_estimate(scale); /// set angles to 0 and U=V=intial_estimate.

        float *r_arr = new float[2*int(R)]; /// R is from 0 to pi, 2*R is from 0 to 2pi.

        int n = (int)N;
        for(int k=0;k<K;k++) {
            float*p_frame = p_data + k*n*n;
            for(int ix=0;ix<2*int(R);ix++) {
                r_arr[ix] = sample_peak(ix,k,p_frame);
            }

            Eigen::Matrix2f cov = get_covariance(r_arr);
            update_estimate_from_covariance(k,cov);
        }

        if( verbose > 0 ) {
            sprintf(filename,"%s/ctf_ellipse_fit",out_dir);
            IO::create_dir(filename);
            float vmin=0;
            float vmax=0;
            float vavg=0;
            float vstd=0;
            Math::get_min_max_avg_std(vmin,vmax,vavg,vstd,p_data,(int)(N*N*K));
            float min_val = fmaxf( vmin, vavg - 10*vstd );
            float max_val = fminf( vmax, vavg + 25*vstd );
            for(int k=0;k<K;k++) {
                float*p_frame = p_data + k*n*n;
                sprintf(filename,"%s/ctf_ellipse_fit/projection_%02d.svg",out_dir,k);
                ctf_ellipse_fit_to_svg(filename,p_frame,(int)N,min_val,max_val,lin_fpix_to_defocus/scale,c_estimate[k].U,c_estimate[k].V,c_estimate[k].angle);
            }
        }

        /// To Angstroms
        float def_sign = is_overfocus ? -1.0f : 1.0f;
        for(int k=0;k<K;k++) {
            c_estimate[k].U *= def_sign*lin_fpix_to_defocus/scale;
            c_estimate[k].V *= def_sign*lin_fpix_to_defocus/scale;
            c_estimate[k].angle *= RAD2DEG;
        }

        delete [] r_arr;
    }

    void estimate_phase_shift(float*c_buffer,GPU::GArrSingle&hermitian_input,const char*out_dir) {
        GPU::GArrSingle  g_acc;
        GPU::GArrSingle  g_wgt;
        GPU::GArrSingle  g_blr;
        GPU::GArrSingle2 g_tmp;
        GPU::GArrSingle2 g_fou;
        GPU::GArrDefocus g_def;
        g_blr.alloc(M*N*K);
        g_tmp.alloc(N*K);
        g_acc.alloc(N*K);
        g_wgt.alloc(N*K);
        g_fou.alloc(N*K);
        g_def.alloc(K);

        GpuFFT::FFT1D_full  fft;
        GpuFFT::IFFT1D_full ifft;
        fft.alloc(N,K);
        ifft.alloc(N,K);

        g_acc.clear();
        g_wgt.clear();

        float*c_ps_raw  = new float[int(N*K)];
        float*c_ps_norm = new float[int(N*K)];

        cudaMemcpy((void*)g_def.ptr, (const void*)c_estimate, sizeof(Defocus)*int(K), cudaMemcpyHostToDevice);

        dim3 blk = GPU::get_block_size_2D();

        int3 siz_ps = make_int3(M,N,K);
        dim3 grd_ps = GPU::calc_grid_size(blk,siz_ps);

        int3 siz_rad = make_int3(N,K,1);
        dim3 grd_rad = GPU::calc_grid_size(blk,siz_rad);

        GpuKernels::conv_gaussian<<<grd_ps,blk>>>(g_blr.ptr,hermitian_input.ptr,0.5000,6.2831,siz_ps);
        GpuKernelsCtf::ctf_radial_normalize_and_avg<<<grd_ps,blk>>>(g_acc.ptr,g_wgt.ptr,g_blr.ptr,g_def.ptr,siz_ps);
        GpuKernels::divide<<<grd_rad,blk>>>(g_acc.ptr,g_wgt.ptr,siz_rad);
        cudaMemcpy((void*)c_ps_raw, (const void*)g_acc.ptr, sizeof(float)*int(N)*int(K), cudaMemcpyDeviceToHost);
        Math::fftshift_1D_batch(c_ps_raw,int(N),int(K));

        GpuKernels::real_to_complex<<<grd_rad,blk>>>(g_tmp.ptr,g_acc.ptr,siz_rad);

        fft.exec(g_fou.ptr,g_tmp.ptr);
        GpuKernels::analytical_signal_1D<<<grd_rad,blk>>>(g_fou.ptr,M,N,1,K);
        ifft.exec(g_tmp.ptr,g_fou.ptr);
        GpuKernels::divide<<<grd_rad,blk>>>(g_tmp.ptr,N,siz_rad);

        GpuKernels::as_amplitude_normalize_clamp<<<grd_rad,blk>>>(g_acc.ptr,g_tmp.ptr,siz_rad);
        cudaMemcpy((void*)c_ps_norm, (const void*)g_acc.ptr, sizeof(float)*int(N)*int(K), cudaMemcpyDeviceToHost);
        Math::fftshift_1D_batch(c_ps_norm,int(N),int(K));

        /// REFINE CTF
        CtfRefiner ctf_refiner(int(M),int(N),int(K),apix,lambda_kv,AC,CS);
        float delta_def,phase_shift,max_score;
        int   max_fpix;

        if( verbose>0 ) {
            sprintf(filename,"%s/ctf_radial_fit",out_dir);
            IO::create_dir(filename);
        }

        for(int k=0;k<int(K);k++) {
            float*p_data  = c_ps_norm + k*int(N);
            float ini_def = (c_estimate[k].U+c_estimate[k].V)/2;
            ctf_refiner.refine_ctf_hybrid(max_score,delta_def,phase_shift,max_fpix,p_data,ini_def,est_phase_shift);
            c_estimate[k].U += delta_def;
            c_estimate[k].V += delta_def;
            c_estimate[k].ph_shft = est_phase_shift ? phase_shift : 0.0f;
            c_estimate[k].score   = max_score;
            c_estimate[k].max_res = N*apix/max_fpix;
            c_estimate[k].Bfactor = 0.0f;
            c_estimate[k].ExpFilt = 0.0f;
            if( verbose>0 ) {
                float p_tmp[int(M)];
                sprintf(filename,"%s/ctf_radial_fit/projection_%02d.svg",out_dir,k);
                SvgCtf svg_ctf(filename,apix);
                svg_ctf.create_grid(0.0f,float(max_fpix),int(N));
                svg_ctf.add_est(ctf_refiner.buf_metric,int(M),true);
                for(int i=0;i<int(M);i++) p_tmp[i] = ctf_refiner.buf_ctf[i]*ctf_refiner.buf_ctf[i];
                svg_ctf.add_fit(p_tmp,int(M));
                for(int i=0;i<int(M);i++) p_tmp[i] = fminf( fmaxf(c_ps_raw[i+k*int(N)]+0.5,0),1.0f);
                svg_ctf.add_avg(p_tmp,int(M));
                svg_ctf.create_legend();
                svg_ctf.create_title(k,ini_def+delta_def);
            }
        }
    }

    void save_fitting(const char*out_dir,float*c_buffer,GPU::GArrSingle&hermitian_input) {
        GPU::GArrSingle  g_vis;
        GPU::GArrDefocus g_def;
        g_def.alloc(K);
        g_vis.alloc(N*N*K);
        g_vis.clear();

        cudaMemcpy((void*)g_def.ptr, (const void*)c_estimate, sizeof(Defocus)*int(K),cudaMemcpyHostToDevice);

        dim3 blk = GPU::get_block_size_2D();

        int3 siz_hermitian = make_int3(M,N,K);
        dim3 grd_hermitian = GPU::calc_grid_size(blk,siz_hermitian);

        int3 siz_full = make_int3(N,N,K);
        dim3 grd_full = GPU::calc_grid_size(blk,siz_full);

        GpuKernelsCtf::vis_copy_data<<<grd_hermitian,blk>>>(g_vis.ptr,hermitian_input.ptr,1.0f/6,0.5f,siz_full,siz_hermitian);
        GpuKernelsCtf::vis_add_ctf<<<grd_full,blk>>>(g_vis.ptr,g_def.ptr,apix,lambda_pi,lambda3_Cs_pi_2,AC,siz_full);

        save_gpu_mrc(c_buffer,g_vis.ptr,N,N,K,out_dir,"ctf_fitting.mrc",0);
    }

    void save_defocus_result(const char*out_dir) {
        sprintf(filename,"%s/defocus.txt",out_dir);
        FILE*fp = fopen(filename,"w");
        for(int k=0;k<K;k++) {
            IO::DefocusIO::write(fp,c_estimate[k]);
        }

        fclose(fp);
    }

};

#endif /// CTF_LINEARIZER_H

