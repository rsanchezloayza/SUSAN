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

/// Consecutive windows below the correlation threshold before the resolution
/// limit is declared: one dip is noise, a run of them is the real cutoff.
#define SNR_MAX_RES_RUN 3
#define ELLIPSE_FIT_ITERS 3
#define ASTIG_SIGMA       3.0     /// series astigmatism kept above this many standard errors
#define SCAN_HALF_WIDTH   2500.0f /// A, defocus scan around each start
#define SCAN_STEP         50.0f   /// A
#define SCAN_FINE_STEP    5.0f    /// A

/// The per-projection SNR is measured over the first CTF peak, from this
/// fraction of the first zero out to the zero itself.  Anchoring to the zero
/// keeps projections comparable when the defocus changes between them.
#define SNR_BAND_INNER_FRAC 0.5f

/// Margin applied to the requested maximum defocus when framing the ellipse
/// fit report, so the outermost fit is not drawn hard against the edge.
#define SVG_DEF_MARGIN 1.2f

/// Upper bound on the fitted basis: CTF^2, its slope, and the background.
#define SNR_FIT_MAX_TERMS (SNR_BAND_BG_ORDER+3)

/// Order of the polynomial that absorbs the background inside that band.
/// Order 0 leaves the background slope in the CTF term and inverts the sign;
/// order 4 starts absorbing the CTF peak itself.
#define SNR_BAND_BG_ORDER 2

class CtfLinearizer1D{

public:
    GPU::GArrSingle  g_input;
    GPU::GArrSingle  g_linear;
    GPU::GArrSingle  g_weight;
    GPU::GArrSingle  g_bufferA;
    GPU::GArrSingle  g_bufferB;
    GPU::GArrSingle2 g_fourier;
    GPU::GArrSingle  g_power;
    GPU::GArrSingle2 g_cplxA;
    GPU::GArrSingle2 g_cplxB;

    GpuFFT::FFT1D       rfft;
    GpuFFT::FFT1D_full  fft_full;
    GpuFFT::IFFT1D_full ifft_full;

    float  *c_linear;
    float  *c_defocus;

    float N;
    float M;
    float K;

    float apix;
    float new_apix;
    float max_nyquist;
    float lin_fpix_to_defocus;

    bool  dechirp_cs;
    float dechirp_coef;

    int   log_level;

    char filename[SUSAN_FILENAME_LENGTH];

    float u0;
    float s0;
    int   k0;

    int   def_ix_min;
    int   def_ix_max;

    CtfLinearizer1D(int gpu_ix, int n, int k, float w_target_apix=3.5f) {

        N = n;
        M = n/2 + 1;
        K = k;

        apix = 1.0;
        new_apix = w_target_apix;

        dechirp_cs   = false;
        dechirp_coef = 0.0f;

        GPU::set_device(gpu_ix);

        rfft.alloc(N,K);
        fft_full.alloc(N,K);
        ifft_full.alloc(N,K);

        g_cplxA.alloc(N*K);
        g_cplxB.alloc(N*K);

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

        log_level   = info->log_level;

        apix = p_tomo->pix_size;
        max_nyquist = apix/new_apix;
        lin_fpix_to_defocus = 2.0f*new_apix*new_apix/lambda;

        dechirp_cs   = info->dechirp_cs;
        dechirp_coef = Math::get_dechirp_coef(lambda,p_tomo->CS,new_apix);

        set_defocus_search_range(info->def_min,info->def_max);
    }

    /// The defocus maps linearly onto the periodogram bin, so def_range bounds
    /// the peak search directly.  An unset range falls back to every bin.
    void set_defocus_search_range(float def_min,float def_max) {
        def_ix_min = int( floorf( def_min/lin_fpix_to_defocus ) );
        def_ix_max = int( ceilf ( def_max/lin_fpix_to_defocus ) );
        if( def_ix_min < 1 ) def_ix_min = 1;
        if( def_ix_max <= def_ix_min || def_ix_max > int(M)-1 ) def_ix_max = int(M)-1;
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
            for(int m=def_ix_min;m<=def_ix_max;m++) {
                if( c_linear[m+k*int(M)] > max_val ) {
                    max_val = c_linear[m+k*int(M)];
                    k0 = k;
                }
                tmp[m] += c_linear[m+k*int(M)];
            }

        max_val = 0;
        for(int m=def_ix_min;m<=def_ix_max;m++) {
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

        /// Per tilt: each projection's own peak (the tracking above only bounds where it may lie).
        for(int k=0;k<K;k++) {
            float*p = c_linear+(k*int(M));
            int m_max = def_ix_min;
            for(int m=def_ix_min;m<=def_ix_max;m++)
                if( p[m] > p[m_max] ) m_max = m;
            float mu = m_max, sg = 0;
            estimate_mu_sigma(mu,sg,p,int(fmax(m_max-4,0)),int(fmin(m_max+4,M-1)));
            c_defocus[k] = lin_fpix_to_defocus*mu;
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

        if( dechirp_cs ) {
            int3 siz_full = make_int3(N,K,1);
            dim3 grd_full = GPU::calc_grid_size(blk_rad,siz_full.x,siz_full.y,siz_full.z);

            GpuKernels::real_to_complex<<<grd_full,blk_rad>>>(g_cplxA.ptr,g_linear.ptr,siz_full);
            fft_full.exec(g_cplxB.ptr,g_cplxA.ptr);
            GpuKernels::analytical_signal_1D<<<grd_full,blk_rad>>>(g_cplxB.ptr,M,N,1,K);
            ifft_full.exec(g_cplxA.ptr,g_cplxB.ptr);
            GpuKernels::divide<<<grd_full,blk_rad>>>(g_cplxA.ptr,N,siz_full);
            GpuKernelsCtf::ctf_dechirp<<<grd_full,blk_rad>>>(g_cplxA.ptr,dechirp_coef,int(N),int(K));
            fft_full.exec(g_cplxB.ptr,g_cplxA.ptr);
            GpuKernels::divide<<<grd_full,blk_rad>>>(g_cplxB.ptr,N,siz_full);
            GpuKernelsCtf::load_ps_half<<<grd_rad,blk_rad>>>(g_power.ptr,g_cplxB.ptr,int(M),int(N),int(K));
            return;
        }

        rfft.exec(g_fourier.ptr,g_linear.ptr);
        GpuKernels::divide<<<grd_rad,blk_rad>>>(g_fourier.ptr,N,siz_rad);
        GpuKernels::load_ps<<<grd_rad,blk_rad>>>(g_power.ptr,g_fourier.ptr,siz_rad);
    }

    void estimate_mu_sigma(float&mu,float&sigma,float*p_data,int ix0,int ix1) {
        float acc   = 0;
        float mu_ws = 0;

        for(int ix=ix0;ix<=ix1;ix++) {
            mu_ws += p_data[ix]*ix;
            acc   += p_data[ix];
        }

        /// Empty/degenerate projection (e.g. a blank frame in the stack):
        /// keep the carried-in mu,sigma so the defocus sweep is not poisoned
        /// with NaN (acc==0 -> 0/0), and the empty projection simply inherits
        /// its neighbour's estimate.
        if( fabsf(acc) < SUSAN_FLOAT_TOL )
            return;

        float l_mu  = mu_ws/acc;
        float l_sig = 0;
        for(int ix=ix0;ix<=ix1;ix++) {
            l_sig += p_data[ix]*((ix-l_mu)*(ix-l_mu));
        }

        mu    = l_mu;
        sigma = sqrtf(l_sig/acc);
    }

    void save_gpu_mrc(single*p_cpu,const single*p_gpu,const int m,const int n,const int k,const char*out_dir,const char*name,const int req_verb) {
        cudaMemcpy((void*)p_cpu, (const void*)p_gpu, sizeof(float)*m*n*k, cudaMemcpyDeviceToHost);
        if( log_level >= req_verb ) {
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
        float*buf_corr;

        float apix;
        float lambda_pi;
        float lambda3_Cs_pi_2;
        float AC;
        float CA;

        CtfRefiner(int m, int n, int k, float apix_in, float lambda_kv, float ac, float cs) {
            M = m;
            N = n;
            K = k;
            buf_ctf    = new float[M];
            buf_corr   = new float[M];

            AC = ac;
            CA = sqrt(1.0-AC*AC);
            apix = apix_in;
            lambda_pi = lambda_kv*M_PI;
            lambda3_Cs_pi_2 = lambda_kv*lambda_kv*lambda_kv*(cs*1e7)*M_PI/2;
        }

        ~CtfRefiner() {
            delete [] buf_ctf;
            delete [] buf_corr;
        }

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

    public:
        /// First CTF zero, in Fourier pixels: the radius where gamma reaches pi.
        /// Both the resolution estimate and the SNR band are anchored to it so
        /// that projections with different defocus are compared at equivalent
        /// positions of the CTF rather than at fixed spatial frequencies.
        int ctf_zero_fpix(float def,float phase,int n_zero) {
            float target = float(n_zero)*(float)M_PI;
            for(int i=1;i<M;i++) {
                float s  = float(i)/(float(N)*apix);
                float s2 = s*s;
                float g  = lambda_pi*def*s2 - lambda3_Cs_pi_2*s2*s2 + phase;
                if( g >= target )
                    return i;
            }
            return 0;
        }

        int first_zero_fpix(float def,float phase) {
            return ctf_zero_fpix(def,phase,1);
        }

        /// Highest frequency at which the fitted CTF still explains the data.
        ///
        /// Inside a window spanning at least one oscillation the radial average
        /// is P = A*CTF^2 + background.  Removing a local linear trend from both
        /// the data and the model leaves only the oscillation, so their
        /// correlation says whether the rings are really there.  Walking
        /// outwards from the first zero, the resolution limit is where the
        /// correlation first falls below thres and stays there.
        int estimate_max_fpix(const float*p_raw,float def,float phase,float thres) {

            ctf_1D(def,phase);

            int hw = M/12;
            if( hw < 4 ) hw = 4;

            for(int i=0;i<M;i++)
                buf_corr[i] = 0.0f;

            for(int i=hw;i<M-hw;i++) {

                int   n  = 2*hw+1;
                int   i0 = i-hw;
                float sx=0,sy=0,sz=0,sxx=0,sxy=0,sxz=0;

                for(int j=0;j<n;j++) {
                    float t = float(j)/float(n-1) - 0.5f;
                    float a = p_raw[i0+j];
                    float b = buf_ctf[i0+j]*buf_ctf[i0+j];
                    sx += t; sy += a; sz += b;
                    sxx += t*t; sxy += t*a; sxz += t*b;
                }

                float den_t = sxx - sx*sx/n;
                if( den_t < SUSAN_FLOAT_TOL ) continue;

                float ma = sy/n, mb = sz/n;
                float ka = (sxy - sx*sy/n)/den_t;
                float kb = (sxz - sx*sz/n)/den_t;

                float saa=0,sbb=0,sab=0;
                for(int j=0;j<n;j++) {
                    float t = float(j)/float(n-1) - 0.5f;
                    float a = p_raw[i0+j]                 - (ma + ka*t);
                    float b = buf_ctf[i0+j]*buf_ctf[i0+j] - (mb + kb*t);
                    saa += a*a; sbb += b*b; sab += a*b;
                }

                if( saa < SUSAN_FLOAT_TOL || sbb < SUSAN_FLOAT_TOL ) continue;

                buf_corr[i] = sab/sqrtf(saa*sbb);
            }

            /// Below the first zero CTF^2 is a single hump: too few oscillations
            /// inside the window for the correlation to mean anything, so the
            /// walk starts there.
            int i_start = first_zero_fpix(def,phase);
            if( i_start < hw ) i_start = hw;

            int last_pass = 0;
            int n_fail    = 0;

            for(int i=i_start;i<M-hw;i++) {
                if( buf_corr[i] > thres ) {
                    last_pass = i;
                    n_fail    = 0;
                }
                else {
                    n_fail++;
                    /// A single dip is noise; a run of them is the limit.
                    if( n_fail >= SNR_MAX_RES_RUN )
                        break;
                }
            }

            return last_pass;
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

    float def_max_angs;

    int  log_level;
    bool est_phase_shift;
    bool est_initial_snr;
    bool dechirp_cs;
    float dechirp_coef;
    float res_thres;

    float   *c_ini_idx;
    float   *c_trk_idx;
    float   *c_rough_def;
    double   astig_amp_px, astig_sigma, astig_theta;
    int      astig_n;
    bool    *c_failed;
    int      verbosity;
    float    def_min_angs;
    float    res_max_angs;
    uint32   astig_type;
    float   *c_prj_wgt;
    bool    *c_wide;
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
        c_trk_idx  = new float[int(K)];
        c_rough_def = new float[int(K)];
        c_failed   = new bool[int(K)];
        c_prj_wgt  = new float[int(K)];
        c_wide     = new bool[int(K)];
        c_estimate = new Defocus[int(K)];
        memset((void*)c_estimate,0,sizeof(Defocus)*int(K));
        for(int k=0;k<int(K);k++) {
            c_prj_wgt[k] = 1.0f;
            c_wide[k]    = false;
            c_failed[k]  = false;
            c_rough_def[k] = 0;
        }
    }

    /// Rough per-projection defocus of pass 1 (sets the background width of the defocus sweep).
    void set_rough_defocus(const float*def) {
        for(int k=0;k<int(K);k++) c_rough_def[k] = fabsf(def[k]);
    }

    ~CtfLinearizer() {

        delete [] c_ini_idx;
        delete [] c_trk_idx;
        delete [] c_rough_def;
        delete [] c_failed;
        delete [] c_prj_wgt;
        delete [] c_wide;
        delete [] c_estimate;
    }

    void load_info(ArgsCTF::Info*info,Tomogram*p_tomo) {

        float lambda = Math::get_lambda(p_tomo->KV);

        apix = p_tomo->pix_size;
        log_level = info->log_level;
        est_phase_shift = info->est_phase_shift;
        est_initial_snr = info->est_initial_snr;
        res_thres       = info->res_thres;
        max_nyquist = apix/new_apix;
        lin_fpix_to_defocus = 2.0f*new_apix*new_apix/lambda;

        def_max_angs = info->def_max;

        def_ix_min = int( floorf( info->def_min/lin_fpix_to_defocus ) );
        def_ix_max = int( ceilf ( info->def_max/lin_fpix_to_defocus ) );
        if( def_ix_min < 1 ) def_ix_min = 1;
        if( def_ix_max <= def_ix_min || def_ix_max > int(M)-1 ) def_ix_max = int(M)-1;

        CS = p_tomo->CS;
        AC = p_tomo->AC;
        CA = sqrt(1.0-AC*AC);
        lambda_kv = Math::get_lambda(p_tomo->KV);
        lambda_pi = lambda_kv*M_PI;
        lambda3_Cs_pi_2 = lambda_kv*lambda_kv*lambda_kv*(CS*1e7)*M_PI/2;

        verbosity    = info->verbosity;
        def_min_angs = info->def_min;
        res_max_angs = info->res_max;
        astig_type   = info->astig_type;
        dechirp_cs   = info->dechirp_cs;
        dechirp_coef = Math::get_dechirp_coef(lambda_kv,CS,new_apix);
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
        if( log_level > 1 ) {
            sprintf(filename,"%s/ps_lin_norm.mrc",out_dir);
            Mrc::write(c_linear,M,K,1,filename,new_apix);
        }
        track_peaks_in_linear_ps(c_linear);
        memcpy(c_trk_idx,c_ini_idx,sizeof(float)*int(K));
        free_peaks_in_linear_ps(c_linear);

        delete [] c_linear;
    }

    /// Per tilt: each projection's own peak in the defocus range, as the weighted centroid over +-4 bins.
    void free_peaks_in_linear_ps(float*c_linear) {
        for(int k=0;k<K;k++) {
            float*p = c_linear+(k*int(M));
            int m_max = def_ix_min;
            for(int m=def_ix_min;m<=def_ix_max;m++)
                if( p[m] > p[m_max] ) m_max = m;
            float mu = m_max, sg = 0;
            estimate_mu_sigma_1D(mu,sg,p,int(fmax(m_max-4,0)),int(fmin(m_max+4,M-1)));
            c_ini_idx[k] = mu;
        }
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

        /// MEAN DEFOCUS (AND PHASE SHIFT) PER PROJECTION
        GPU::GArrSingle g_1d;
        g_1d.alloc(M*N*K);
        load_remove_bg_normalize_var(g_1d,input);
        sweep_defocus(g_1d,out_dir);

        /// FAILED PER-TILT ESTIMATES: no ring band past the first zero follows the fitted CTF (max_res 0) -> prj_wgt 0.
        for(int k=0;k<K;k++) c_failed[k] = !(c_estimate[k].max_res > 0);

        /// FLAG EMPTY PROJECTIONS (blank frames in the stack)
        mark_empty_projections(input,p_tomo);

        /// INITIAL PER-PROJECTION WEIGHT FROM THE FIRST CTF PEAK
        if( est_initial_snr )
            estimate_initial_snr(input,out_dir);

        for(int k=0;k<K;k++)
            if( c_failed[k] ) c_prj_wgt[k] = 0.0f;

        /// SAVE RESULTS
        save_fitting(out_dir,c_buffer,g_input);
        save_defocus_result(out_dir);

        delete [] c_buffer;
    }

    void estimate_initial_snr(float*input,const char*out_dir) {

        /// The background is a term of the model and the denominator of the
        /// ratio, so this pass runs on the raw average, not on the
        /// background-removed one used for the defocus refinement.
        GPU::GArrSingle  g_raw;
        GPU::GArrSingle  g_acc;
        GPU::GArrSingle  g_wgt;
        GPU::GArrDefocus g_def;

        g_raw.alloc(M*N*K);
        g_acc.alloc(N*K);
        g_wgt.alloc(N*K);
        g_def.alloc(K);
        g_acc.clear();
        g_wgt.clear();

        cudaMemcpy((void*)g_raw.ptr,(const void*)input,sizeof(float)*int(M)*int(N)*int(K),cudaMemcpyHostToDevice);
        cudaMemcpy((void*)g_def.ptr,(const void*)c_estimate,sizeof(Defocus)*int(K),cudaMemcpyHostToDevice);

        dim3 blk = GPU::get_block_size_2D();
        int3 siz_ps  = make_int3(M,N,K);
        dim3 grd_ps  = GPU::calc_grid_size(blk,siz_ps);
        int3 siz_rad = make_int3(N,K,1);
        dim3 grd_rad = GPU::calc_grid_size(blk,siz_rad);

        GpuKernelsCtf::ctf_radial_normalize_and_avg<<<grd_ps,blk>>>(g_acc.ptr,g_wgt.ptr,g_raw.ptr,g_def.ptr,siz_ps);
        GpuKernels::divide<<<grd_rad,blk>>>(g_acc.ptr,g_wgt.ptr,siz_rad);

        float*c_prof = new float[int(N)*int(K)];
        cudaMemcpy((void*)c_prof,(const void*)g_acc.ptr,sizeof(float)*int(N)*int(K),cudaMemcpyDeviceToHost);
        Math::fftshift_1D_batch(c_prof,int(N),int(K));

        CtfRefiner refiner(int(M),int(N),int(K),apix,lambda_kv,AC,CS);

        for(int k=0;k<int(K);k++) {

            float def = (c_estimate[k].U+c_estimate[k].V)/2;
            def = fabsf(def);   /// overfocus stores negative defocus

            if( def < SUSAN_FLOAT_TOL ) {
                c_prj_wgt[k] = 0.0f;
                continue;
            }

            int r1 = refiner.ctf_zero_fpix(def,c_estimate[k].ph_shft,1);
            int r2 = refiner.ctf_zero_fpix(def,c_estimate[k].ph_shft,2);
            int r0 = (int)(SNR_BAND_INNER_FRAC*r1);

            if( r1 <= 0 || r1 >= int(M) || (r1-r0) < (SNR_BAND_BG_ORDER+4) ) {
                c_prj_wgt[k] = 0.0f;
                continue;
            }

            /// Two peaks whenever the second zero is inside the profile; the
            /// oscillation is what separates the signal from the background.
            /// At low defocus it falls beyond the fitted range, and the fit
            /// falls back to the first peak alone.
            c_wide[k] = ( r2 > r1 && r2 < int(M) && (r2-r0) >= (SNR_BAND_BG_ORDER+6) );
            int r_end = c_wide[k] ? r2 : r1;

            refiner.ctf_1D(def,c_estimate[k].ph_shft);
            c_prj_wgt[k] = fit_band_ssnr(c_prof+k*int(N),refiner.buf_ctf,r0,r_end,r1,c_wide[k]);
        }

        for(int k=0;k<int(K);k++)
            if( c_failed[k] ) c_prj_wgt[k] = 0.0f;
        sqrt_prj_wgt();
        normalize_prj_wgt();

        if( log_level > 0 )
            save_snr_report(out_dir);

        delete [] c_prof;
    }

    /// Least squares fit of
    ///     P(r) = (A0 + A1*t)*CTF^2(r) + sum_p b_p*t^p
    /// over [r0,r_end), with t normalized to [0,1] across that range.
    ///
    /// The amplitude slope is used only when the band spans two CTF peaks.  A
    /// single amplitude cannot match both once the envelope has decayed
    /// between them, and that mismatch is structured in r, so the background
    /// polynomial absorbs it and the estimate degrades; the slope removes it.
    /// Over one peak the slope is collinear with the polynomial and is left
    /// out.
    ///
    /// The ratio is always evaluated over the first peak, [r0,r1): the
    /// amplitude at the centre of that stretch over the mean background on it.
    float fit_band_ssnr(const float*p_prof,const float*p_ctf,int r0,int r_end,int r1,bool use_slope) {

        const int n_bg = SNR_BAND_BG_ORDER+1;
        const int nb   = 1 + (use_slope?1:0) + n_bg;
        const int i_bg = nb - n_bg;              /// first background column

        int n    = r_end-r0;
        int n_pk = r1-r0;

        if( n < nb+2 || n_pk < 2 )
            return 0.0f;

        double AtA[SNR_FIT_MAX_TERMS*SNR_FIT_MAX_TERMS];
        double Atb[SNR_FIT_MAX_TERMS];
        double basis[SNR_FIT_MAX_TERMS];

        for(int i=0;i<nb*nb;i++) AtA[i] = 0.0;
        for(int i=0;i<nb;i++)    Atb[i] = 0.0;

        for(int i=0;i<n;i++) {
            float  t  = float(i)/float(n-1);
            double c2 = p_ctf[r0+i]*p_ctf[r0+i];

            basis[0] = c2;
            if( use_slope ) basis[1] = c2*t;

            double tp = 1.0;
            for(int q=0;q<n_bg;q++) { basis[i_bg+q] = tp; tp *= t; }

            double y = p_prof[r0+i];
            for(int a=0;a<nb;a++) {
                Atb[a] += basis[a]*y;
                for(int b=0;b<nb;b++)
                    AtA[a*nb+b] += basis[a]*basis[b];
            }
        }

        /// Gauss-Jordan with partial pivoting: the system is at most 5x5.
        double sol[SNR_FIT_MAX_TERMS];
        for(int i=0;i<nb;i++) sol[i] = Atb[i];
        for(int c=0;c<nb;c++) {
            int piv = c;
            for(int rr=c+1;rr<nb;rr++)
                if( fabs(AtA[rr*nb+c]) > fabs(AtA[piv*nb+c]) ) piv = rr;
            if( fabs(AtA[piv*nb+c]) < 1e-12 ) return 0.0f;
            if( piv != c ) {
                for(int j=0;j<nb;j++) std::swap(AtA[c*nb+j],AtA[piv*nb+j]);
                std::swap(sol[c],sol[piv]);
            }
            double d = AtA[c*nb+c];
            for(int j=0;j<nb;j++) AtA[c*nb+j] /= d;
            sol[c] /= d;
            for(int rr=0;rr<nb;rr++) {
                if( rr==c ) continue;
                double f = AtA[rr*nb+c];
                for(int j=0;j<nb;j++) AtA[rr*nb+j] -= f*AtA[c*nb+j];
                sol[rr] -= f*sol[c];
            }
        }

        /// Amplitude at the centre of the first peak, background averaged on it.
        double t_mid = 0.0;
        double bg    = 0.0;
        for(int i=0;i<n_pk;i++) {
            float  t  = float(i)/float(n-1);
            double tp = 1.0, v = 0.0;
            for(int q=0;q<n_bg;q++) { v += sol[i_bg+q]*tp; tp *= t; }
            bg    += v;
            t_mid += t;
        }
        bg    /= n_pk;
        t_mid /= n_pk;

        double A = sol[0] + ( use_slope ? sol[1]*t_mid : 0.0 );

        if( bg <= 0.0 || A <= 0.0 )
            return 0.0f;

        return (float)(A/bg);
    }

    void normalize_prj_wgt() {
        float max_wgt = 0;
        for(int k=0;k<int(K);k++)
            max_wgt = fmaxf(max_wgt,c_prj_wgt[k]);

        if( max_wgt < SUSAN_FLOAT_TOL ) {
            /// No usable first peak anywhere: fall back to uniform weights.
            for(int k=0;k<int(K);k++)
                c_prj_wgt[k] = 1.0f;
            return;
        }

        for(int k=0;k<int(K);k++)
            c_prj_wgt[k] = c_prj_wgt[k]/max_wgt;
    }

    /// prj_w enters linearly (sum w c X / sum w c^2 and the weighted CC average); for projections normalized to
    /// unit noise the optimum is the amplitude ratio, sqrt of the power SNR that fit_band_ssnr returns.
    void sqrt_prj_wgt() {
        for(int k=0;k<int(K);k++)
            c_prj_wgt[k] = sqrtf(fmaxf(c_prj_wgt[k],0.0f));
    }

    void save_snr_report(const char*out_dir) {
        sprintf(filename,"%s/initial_snr.txt",out_dir);
        FILE*fp = fopen(filename,"w");
        fprintf(fp,"# proj   defocus    zero_1  zero_2       band_fpix   mode   max_res     prj_wgt\n");
        CtfRefiner refiner(int(M),int(N),int(K),apix,lambda_kv,AC,CS);
        for(int k=0;k<int(K);k++) {
            float def = (c_estimate[k].U+c_estimate[k].V)/2;
            def = fabsf(def);
            int r1 = (def>SUSAN_FLOAT_TOL) ? refiner.ctf_zero_fpix(def,c_estimate[k].ph_shft,1) : 0;
            int r2 = (def>SUSAN_FLOAT_TOL) ? refiner.ctf_zero_fpix(def,c_estimate[k].ph_shft,2) : 0;
            int r0 = (int)(SNR_BAND_INNER_FRAC*r1);
            fprintf(fp,"%6d  %10.1f  %6d  %6d   %6d-%-6d  %6s  %8.2f  %10.6f\n",
                    k,def,r1,r2,r0,(c_wide[k]?r2:r1),(c_wide[k]?"2peak":"1peak"),
                    c_estimate[k].max_res,c_prj_wgt[k]);
        }
        fclose(fp);
    }

    /// A blank frame in the stack yields an all-zero averaged power spectrum
    /// slice.  Its defocus is only a carried-over neighbour estimate, so we
    /// flag it as invalid (score=0, max_res=0).  max_res=0 is already treated
    /// as "excluded" by the reconstruction/aligner band-pass setup.
    void mark_empty_projections(const float*input,Tomogram*p_tomo) {
        long slice = (long)M*(long)N;
        for(int k=0;k<K;k++) {
            const float*p_slice = input + (long)k*slice;
            double acc = 0.0;
            for(long i=0;i<slice;i++)
                acc += fabs(p_slice[i]);
            if( acc < SUSAN_FLOAT_TOL ) {
                /// Disable the projection: zero its weight and its (otherwise
                /// NaN / meaningless carried-over) defocus estimate.
                p_tomo->w[k] = 0.0f;
                c_estimate[k].U       = 0.0f;
                c_estimate[k].V       = 0.0f;
                c_estimate[k].angle   = 0.0f;
                c_estimate[k].ph_shft = 0.0f;
                c_estimate[k].Bfactor = 0.0f;
                c_estimate[k].ExpFilt = 0.0f;
                c_estimate[k].max_res = 0.0f;
                c_estimate[k].score   = 0.0f;
                c_prj_wgt[k]          = 0.0f;
            }
        }
    }


    /// mean U, V and the standard deviation of the mean defocus over the projections with weight
    void summary(float&du,float&dv,float&spread) {
        double su=0, sv=0, sd=0, sd2=0;
        int n = 0;
        for(int k=0;k<K;k++) {
            if( !(c_prj_wgt[k] > 0) ) continue;
            double d = 0.5*(c_estimate[k].U+c_estimate[k].V);
            su += c_estimate[k].U; sv += c_estimate[k].V; sd += d; sd2 += d*d; n++;
        }
        du = dv = spread = 0;
        if( n > 0 ) {
            du = su/n; dv = sv/n;
            spread = sqrt(fmax(sd2/n - (sd/n)*(sd/n),0.0));
        }
    }

protected:
    void save_cpu_mrc(single*p_cpu,const int m,const int n,const int k,const char*out_dir,const char*name,const int req_verb) {
        if( log_level >= req_verb ) {
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
        float acc   = 0;
        float mu_ws = 0;

        for(int ix=ix0;ix<=ix1;ix++) {
            mu_ws += p_data[ix]*ix;
            acc   += p_data[ix];
        }

        /// Empty/degenerate projection (e.g. a blank frame in the stack):
        /// keep the carried-in mu,sigma so the peak tracking is not poisoned
        /// with NaN (acc==0 -> 0/0), and the empty projection simply inherits
        /// its neighbour's estimate.
        if( fabsf(acc) < SUSAN_FLOAT_TOL )
            return;

        float l_mu  = mu_ws/acc;
        float l_sig = 0;
        for(int ix=ix0;ix<=ix1;ix++) {
            l_sig += p_data[ix]*((ix-l_mu)*(ix-l_mu));
        }

        mu    = l_mu;
        sigma = sqrtf(l_sig/acc);
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
            for(int m=def_ix_min;m<=def_ix_max;m++) {
                if( c_linear[m+k*int(M)] > max_val ) {
                    max_val = c_linear[m+k*int(M)];
                    k0 = k;
                }
                p_avg[m] += c_linear[m+k*int(M)];
            }

        max_val = 0;
        for(int m=def_ix_min;m<=def_ix_max;m++) {
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

        if( dechirp_cs ) {
            GPU::GArrSingle2    g_cplxA;
            GPU::GArrSingle2    g_cplxB;
            GpuFFT::FFT1D_full  fft_full;
            GpuFFT::IFFT1D_full ifft_full;

            g_cplxA.alloc(N*K);
            g_cplxB.alloc(N*K);
            fft_full.alloc(N,K);
            ifft_full.alloc(N,K);

            GpuKernels::real_to_complex<<<grd_rad,blk>>>(g_cplxA.ptr,g_acc.ptr,siz_rad);
            fft_full.exec(g_cplxB.ptr,g_cplxA.ptr);
            GpuKernels::analytical_signal_1D<<<grd_rad,blk>>>(g_cplxB.ptr,M,N,1,K);
            ifft_full.exec(g_cplxA.ptr,g_cplxB.ptr);
            GpuKernels::divide<<<grd_rad,blk>>>(g_cplxA.ptr,N,siz_rad);
            GpuKernelsCtf::ctf_dechirp<<<grd_rad,blk>>>(g_cplxA.ptr,dechirp_coef,int(N),int(K));
            fft_full.exec(g_cplxB.ptr,g_cplxA.ptr);
            GpuKernels::divide<<<grd_rad,blk>>>(g_cplxB.ptr,N,siz_rad);
            GpuKernelsCtf::load_ps_half<<<grd_lin,blk>>>(g_ps.ptr,g_cplxB.ptr,int(M),int(N),int(K));
            return;
        }

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

    /// Input of the defocus sweep: 2D background removal with sigma = P1/3.5 per projection, P1 the Thon ring period
    /// at the first zero (s1/2 in frequency, s1 = 1/sqrt(lambda d), d the rough defocus of pass 1), then zero mean and
    /// unit std per slice. The ellipse keeps the sigma-4 px removal of load_remove_bg_normalize.
    void load_remove_bg_normalize_var(GPU::GArrSingle&rslt,float*input) {
        GPU::GArrSingle g_bufferA;
        GPU::GArrSingle g_bufferB;
        GPU::GArrSingle g_sigma;
        g_bufferA.alloc(M*N*K);
        g_bufferB.alloc(M*N*K);
        g_sigma.alloc(K);

        std::vector<float> d((size_t)K), pos;
        for(int k=0;k<K;k++) { d[k] = c_rough_def[k]; if( d[k] > 0 ) pos.push_back(d[k]); }
        float d_med = 20000.0f;
        if( !pos.empty() ) { std::sort(pos.begin(),pos.end()); d_med = (pos.size()%2) ? pos[pos.size()/2] : 0.5f*(pos[pos.size()/2-1]+pos[pos.size()/2]); }
        std::vector<float> sig((size_t)K);
        for(int k=0;k<K;k++) {
            double dk = (d[k] > 0) ? d[k] : d_med;
            double s1 = 1.0/sqrt((double)lambda_kv*dk);
            sig[k] = fmaxf(roundf(float(s1/2*double(N)*apix/3.5*2))/2,1.0f);
        }
        cudaMemcpy( (void*)g_sigma.ptr, (const void*)sig.data(), sizeof(float)*int(K), cudaMemcpyHostToDevice);
        cudaMemcpy( (void*)rslt.ptr, (const void*)(input), sizeof(float)*M*N*K, cudaMemcpyHostToDevice);

        int3 siz = make_int3(M,N,K);
        dim3 blk = GPU::get_block_size_2D();
        dim3 grd = GPU::calc_grid_size(blk,siz.x,siz.y,siz.z);
        GpuKernels::conv_gauss_1D_Y_var<<<grd,blk>>>(g_bufferA.ptr,rslt.ptr,g_sigma.ptr,siz);
        GpuKernels::conv_gauss_1D_X_var<<<grd,blk>>>(g_bufferB.ptr,g_bufferA.ptr,g_sigma.ptr,siz);
        GpuKernels::substract<<<grd,blk>>>(rslt.ptr,g_bufferB.ptr,siz);
        g_bufferA.clear();
        g_bufferB.clear();
        GpuKernels::get_avg_std<<<grd,blk>>>(g_bufferA.ptr,g_bufferB.ptr,rslt.ptr,siz);
        GpuKernels::zero_avg_one_std<<<grd,blk>>>(rslt.ptr,g_bufferA.ptr,g_bufferB.ptr,siz);
    }

    /// Local correlations (estimate_max_fpix's windows of 2hw+1, linear trend removed from data and model) of a
    /// radial profile with CTF^2(def,phase), window centres i0..i1 (inclusive); returns their mean.
    double band_corr(CtfRefiner&ref,const float*raw,double def,double phase,int i0,int i1,double*c_out=0) {
        int hw = int(M)/12; if( hw < 4 ) hw = 4;
        int n = 2*hw+1;
        if( i0 < hw ) i0 = hw;
        if( i1 > int(M)-hw-1 ) i1 = int(M)-hw-1;
        double tt = 0;
        for(int j=0;j<n;j++) { double t = double(j)/(n-1)-0.5; tt += t*t; }
        double df = 1.0/(double(N)*apix);
        double acc = 0; int cnt = 0;
        std::vector<double> xw(n), yw(n);
        for(int i=i0;i<=i1;i++) {
            double my=0, mx=0, ty=0, tx=0;
            for(int j=0;j<n;j++) {
                double t = double(j)/(n-1)-0.5;
                double s = (i-hw+j)*df, s2 = s*s;
                double g = (double)ref.lambda_pi*def*s2 - (double)ref.lambda3_Cs_pi_2*s2*s2 + phase;
                double c = -((double)ref.CA*sin(g)+(double)ref.AC*cos(g));
                xw[j] = c*c; yw[j] = raw[i-hw+j];
                mx += xw[j]; my += yw[j]; tx += t*xw[j]; ty += t*yw[j];
            }
            mx /= n; my /= n; tx /= tt; ty /= tt;
            double sxy=0, sxx=0, syy=0;
            for(int j=0;j<n;j++) {
                double t = double(j)/(n-1)-0.5;
                double rx = xw[j]-mx-tx*t, ry = yw[j]-my-ty*t;
                sxy += rx*ry; sxx += rx*rx; syy += ry*ry;
            }
            double c = (sxx > 0 && syy > 0) ? sxy/sqrt(sxx*syy) : 0;
            if( c_out ) c_out[i] = c;
            acc += c; cnt++;
        }
        return (cnt > 0) ? acc/cnt : -2.0;
    }

    /// Mean defocus (and phase shift) per projection, low to high frequency: the local ring correlation averaged over
    /// the first rings (first to third zero) is scanned over +-SCAN_HALF_WIDTH around the ellipse mean defocus and the
    /// tracked start (SCAN_STEP), and with est_phase_shift over the phase 0..pi (0.1 rad); then the band is extended
    /// outward while the correlation at the optimum stays > 0.3 (3 windows below it end it; not past res_max) and
    /// (d, phase) are re-scanned on it (+-500 A in 10 A, +-0.15 rad in 0.01 rad); twice. max_res at the result.
    void sweep_defocus(GPU::GArrSingle&g_1d,const char*out_dir) {
        GPU::GArrSingle  g_acc, g_wgt, g_blr;
        GPU::GArrDefocus g_def;
        g_blr.alloc(M*N*K);
        g_acc.alloc(N*K);
        g_wgt.alloc(N*K);
        g_def.alloc(K);
        g_acc.clear();
        g_wgt.clear();
        float*c_raw = new float[int(N*K)];
        cudaMemcpy((void*)g_def.ptr, (const void*)c_estimate, sizeof(Defocus)*int(K), cudaMemcpyHostToDevice);
        dim3 blk = GPU::get_block_size_2D();
        int3 siz_ps = make_int3(M,N,K);
        dim3 grd_ps = GPU::calc_grid_size(blk,siz_ps);
        int3 siz_rad = make_int3(N,K,1);
        dim3 grd_rad = GPU::calc_grid_size(blk,siz_rad);
        GpuKernels::conv_gaussian<<<grd_ps,blk>>>(g_blr.ptr,g_1d.ptr,0.5000,6.2831,siz_ps);
        GpuKernelsCtf::ctf_radial_normalize_and_avg<<<grd_ps,blk>>>(g_acc.ptr,g_wgt.ptr,g_blr.ptr,g_def.ptr,siz_ps);
        GpuKernels::divide<<<grd_rad,blk>>>(g_acc.ptr,g_wgt.ptr,siz_rad);
        cudaMemcpy((void*)c_raw, (const void*)g_acc.ptr, sizeof(float)*int(N)*int(K), cudaMemcpyDeviceToHost);
        Math::fftshift_1D_batch(c_raw,int(N),int(K));

        CtfRefiner ref(int(M),int(N),int(K),apix,lambda_kv,AC,CS);
        int hw = int(M)/12; if( hw < 4 ) hw = 4;
        int cap = int(double(N)*apix/res_max_angs);
        float lo = (def_min_angs > 0) ? fmaxf(def_min_angs,SCAN_STEP) : SCAN_STEP;
        float hi = (def_max_angs > def_min_angs) ? def_max_angs : 1e9f;
        std::vector<double> cbuf(int(M),0.0);
        if( log_level > 0 ) {
            sprintf(filename,"%s/ctf_radial_fit",out_dir);
            IO::create_dir(filename);
        }
        for(int k=0;k<int(K);k++) {
            const float*raw = c_raw + k*int(N);
            float ini = (c_estimate[k].U+c_estimate[k].V)/2;
            double sgn = (ini < 0) ? -1.0 : 1.0;
            double starts[2] = { fabs(ini), double(lin_fpix_to_defocus)*c_trk_idx[k] };
            std::vector<double> grid;
            for(int c=0;c<2;c++) {
                if( !(starts[c] > 0) ) continue;
                for(double d=starts[c]-SCAN_HALF_WIDTH; d<=starts[c]+SCAN_HALF_WIDTH+SCAN_STEP/2; d+=SCAN_STEP)
                    if( d >= lo && d <= hi ) grid.push_back(round(d*1000)/1000);
            }
            std::sort(grid.begin(),grid.end());
            grid.erase(std::unique(grid.begin(),grid.end()),grid.end());
            if( grid.empty() ) grid.push_back(fmax(starts[0],(double)lo));
            int n_ph = est_phase_shift ? 32 : 1;
            double best = -3, d = grid[0], ph = 0;
            for(int ip=0;ip<n_ph;ip++) {
                double p0 = 0.1*ip;
                for(double x : grid) {
                    int i0 = ref.ctf_zero_fpix(float(x),float(p0),1);
                    int i3 = std::max(ref.ctf_zero_fpix(float(x),float(p0),3),std::max(i0,hw)+4);
                    double v = band_corr(ref,raw,sgn*x,p0,std::max(i0,hw),std::min(i3,cap));
                    if( v > best ) { best = v; d = x; ph = p0; }
                }
            }
            for(int pass=0;pass<2;pass++) {
                int i0 = std::max(ref.ctf_zero_fpix(float(d),float(ph),1),hw);
                std::fill(cbuf.begin(),cbuf.end(),0.0);
                band_corr(ref,raw,sgn*d,ph,hw,int(M)-hw-1,cbuf.data());
                int end = i0, fails = 0;
                for(int i=i0;i<=int(M)-hw-1;i++) {
                    if( i > cap ) break;
                    if( cbuf[i] > 0.3 ) { end = i; fails = 0; }
                    else if( ++fails >= 3 ) break;
                }
                end = std::min(std::max(end,ref.ctf_zero_fpix(float(d),float(ph),3)),cap);
                best = -3;
                double bd = d, bp = ph;
                int np = est_phase_shift ? 31 : 1;
                for(int ip=0;ip<np;ip++) {
                    double p0 = est_phase_shift ? ph - 0.15 + 0.01*ip : 0.0;
                    for(double x=d-500; x<=d+500+5; x+=10) {
                        if( x < lo ) continue;
                        int j0 = std::max(ref.ctf_zero_fpix(float(x),float(p0),1),hw);
                        double v = band_corr(ref,raw,sgn*x,p0,j0,end);
                        if( v > best ) { best = v; bd = x; bp = p0; }
                    }
                }
                d = bd; ph = bp;
            }
            ph = fmod(fmod(ph,M_PI)+M_PI,M_PI);
            float dd = float(sgn*d) - ini;
            c_estimate[k].U += dd;
            c_estimate[k].V += dd;
            c_estimate[k].ph_shft = est_phase_shift ? float(ph) : 0.0f;
            c_estimate[k].score   = float(best);
            c_estimate[k].Bfactor = 0.0f;
            c_estimate[k].ExpFilt = 0.0f;
            int max_fpix = ref.estimate_max_fpix(raw,float(sgn*d),c_estimate[k].ph_shft,res_thres);
            c_estimate[k].max_res = (max_fpix>0) ? N*apix/max_fpix : 0.0f;
            if( log_level > 0 ) {
                float p_tmp[int(M)];
                sprintf(filename,"%s/ctf_radial_fit/projection_%02d.svg",out_dir,k);
                SvgCtf svg_ctf(filename,apix);
                svg_ctf.create_grid(0.0f,float(max_fpix),int(N));
                for(int i=0;i<int(M);i++) p_tmp[i] = ref.buf_corr[i];
                svg_ctf.add_est(p_tmp,int(M),true);
                for(int i=0;i<int(M);i++) p_tmp[i] = ref.buf_ctf[i]*ref.buf_ctf[i];
                svg_ctf.add_fit(p_tmp,int(M));
                for(int i=0;i<int(M);i++) p_tmp[i] = fminf( fmaxf(raw[i]+0.5,0),1.0f);
                svg_ctf.add_avg(p_tmp,int(M));
                svg_ctf.add_corr(ref.buf_corr,int(M),res_thres);
                svg_ctf.create_legend();
                svg_ctf.create_title(k,float(sgn*d));
            }
        }
        delete [] c_raw;
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
        if( dechirp_cs )
            GpuKernelsCtf::ctf_dechirp<<<grd_polar,blk>>>(polar_complex.ptr,dechirp_coef,int(N),int(R*K));
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

    /// Ring radius of the current estimate in direction ix: m + (U-V)/2 cos 2(theta - angle) (fit_harmonic_series).
    float ring_radius(int ix,int k) {
        float a = (ang_step*ix) - c_estimate[k].angle;
        return (c_estimate[k].U+c_estimate[k].V)/2 + (c_estimate[k].U-c_estimate[k].V)/2*cosf(2*a);
    }

    float sample_peak(int ix,int k,float*p_frame,float search_range=3) {
        int   n = (int)N;
        float rad = ring_radius(ix,k);
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

    /// Ring of the PS of the linearized spectrum: its radius is proportional to the defocus in that direction,
    /// r(theta) = m + a cos 2theta + b sin 2theta (linear least squares; no ellipse approximation). m per projection;
    /// (a,b), the objective lens astigmatism, shared by the series: weighted mean of the per-projection (a,b)
    /// (weights 1/variance, projections more than 3 sigma away dropped), set to 0 below ASTIG_SIGMA standard
    /// errors. ELLIPSE_FIT_ITERS passes, the sampling re-centred on the last curve. Writes U, V (PS px) and the angle
    /// of the largest radius (PS frame) into c_estimate.
    void fit_harmonic_series(float*p_data,float*r_arr) {
        int n = (int)N;
        int nr = 2*int(R);
        Eigen::Matrix3d XtX = Eigen::Matrix3d::Zero();
        for(int ix=0;ix<nr;ix++) {
            double a = (double)ang_step*ix;
            Eigen::Vector3d x(1.0,cos(2*a),sin(2*a));
            XtX += x*x.transpose();
        }
        Eigen::Matrix3d XtX_inv = XtX.inverse();
        double*coef = new double[3*int(K)];
        double*var  = new double[int(K)];
        double sa = 0, sb = 0;
        for(int k=0;k<K;k++) {
            c_estimate[k].U = c_estimate[k].V = 2*c_ini_idx[k];
            c_estimate[k].angle = 0;
        }
        for(int it=0;it<ELLIPSE_FIT_ITERS;it++) {
            for(int k=0;k<K;k++) {
                float*p_frame = p_data + k*n*n;
                Eigen::Vector3d Xty = Eigen::Vector3d::Zero();
                for(int ix=0;ix<nr;ix++) {
                    r_arr[ix] = sample_peak(ix,k,p_frame);
                    double a = (double)ang_step*ix;
                    Xty += Eigen::Vector3d(1.0,cos(2*a),sin(2*a))*r_arr[ix];
                }
                Eigen::Vector3d c = XtX_inv*Xty;
                double rss = 0;
                for(int ix=0;ix<nr;ix++) {
                    double a = (double)ang_step*ix;
                    double e = r_arr[ix] - (c(0)+c(1)*cos(2*a)+c(2)*sin(2*a));
                    rss += e*e;
                }
                /// neighbouring directions read overlapping pixels: AR(1) correction of the residual variance
                double r0 = 0, r1 = 0, e_prev = 0, e_first = 0;
                for(int ix=0;ix<nr;ix++) {
                    double a = (double)ang_step*ix;
                    double e = r_arr[ix] - (c(0)+c(1)*cos(2*a)+c(2)*sin(2*a));
                    r0 += e*e;
                    if( ix > 0 ) r1 += e*e_prev; else e_first = e;
                    e_prev = e;
                }
                r1 += e_prev*e_first;
                double rho = (r0 > 0) ? fmin(fmax(r1/r0,0.0),0.95) : 0.0;
                coef[3*k] = c(0); coef[3*k+1] = c(1); coef[3*k+2] = c(2);
                var[k] = rss/fmax(nr-3,1)*0.5*(XtX_inv(1,1)+XtX_inv(2,2))*(1+rho)/(1-rho);
            }
            if( astig_type != ASTIG_SERIES ) {
                /// 0 none, 3 per projection, 1 magnitude: one U-V per series (sum (|z|^2 - 2var)/var / sum 1/var,
                /// tested against chi^2_2 per projection), the angle of each projection where its |z| >= 2 sigma.
                double amp = 0, stat = 0;
                int n_ok = 0, n_ang = 0;
                if( astig_type == ASTIG_MAGNITUDE ) {
                    double s_z = 0, s_w = 0, s_q = 0;
                    for(int k=0;k<K;k++) {
                        if( !(var[k] > 0) || !std::isfinite(var[k]) || !(coef[3*k] > 0) ) continue;
                        double z2 = coef[3*k+1]*coef[3*k+1]+coef[3*k+2]*coef[3*k+2];
                        s_q += z2/var[k]; s_z += (z2-2*var[k])/var[k]; s_w += 1/var[k]; n_ok++;
                    }
                    stat = (n_ok > 0) ? (s_q - 2*n_ok)/(2*sqrt(double(n_ok))) : 0;
                    amp  = (s_w > 0) ? sqrt(fmax(s_z/s_w,0.0)) : 0;
                    if( stat < ASTIG_SIGMA ) amp = 0;
                }
                for(int k=0;k<K;k++) {
                    double am = 0, th = 0;
                    double z = sqrt(coef[3*k+1]*coef[3*k+1]+coef[3*k+2]*coef[3*k+2]);
                    bool ok = (var[k] > 0) && std::isfinite(var[k]) && (coef[3*k] > 0);
                    if( astig_type == ASTIG_PROJECTION && ok ) { am = z; th = 0.5*atan2(coef[3*k+2],coef[3*k+1]); }
                    if( astig_type == ASTIG_MAGNITUDE && ok && z >= 2*sqrt(var[k]) ) { am = amp; th = 0.5*atan2(coef[3*k+2],coef[3*k+1]); if( am > 0 ) n_ang++; }
                    c_estimate[k].U = coef[3*k] + am;
                    c_estimate[k].V = coef[3*k] - am;
                    c_estimate[k].angle = th;
                }
                astig_amp_px = amp; astig_sigma = stat; astig_theta = 0; astig_n = (astig_type == ASTIG_MAGNITUDE) ? n_ang : 0;
                continue;
            }
            /// series astigmatism
            bool*use = new bool[int(K)];
            for(int k=0;k<K;k++) use[k] = (var[k] > 0) && std::isfinite(var[k]) && (coef[3*k] > 0);
            double sw = 0;
            for(int rep=0;rep<3;rep++) {
                sw = 0; sa = 0; sb = 0;
                for(int k=0;k<K;k++) if( use[k] ) { sw += 1/var[k]; sa += coef[3*k+1]/var[k]; sb += coef[3*k+2]/var[k]; }
                if( sw <= 0 ) break;
                sa /= sw; sb /= sw;
                for(int k=0;k<K;k++) {
                    if( !use[k] ) continue;
                    double da = coef[3*k+1]-sa, db = coef[3*k+2]-sb;
                    if( sqrt((da*da+db*db)/var[k]) > 3 ) use[k] = false;
                }
            }
            double se  = (sw > 0) ? 1/sqrt(sw) : 0;
            double amp = sqrt(sa*sa+sb*sb);
            astig_amp_px = amp; astig_sigma = (se > 0) ? amp/se : 0; astig_theta = 0.5*atan2(sb,sa);
            astig_n = 0;
            for(int k=0;k<K;k++) if( use[k] ) astig_n++;
            if( astig_sigma < ASTIG_SIGMA ) { sa = 0; sb = 0; }
            delete [] use;
            double am = sqrt(sa*sa+sb*sb), th = 0.5*atan2(sb,sa);
            for(int k=0;k<K;k++) {
                c_estimate[k].U = coef[3*k] + am;
                c_estimate[k].V = coef[3*k] - am;
                c_estimate[k].angle = th;
            }
        }
        delete [] coef;
        delete [] var;
    }

    void fit_ellipsoid(float*p_data,const char*out_dir,bool is_overfocus,int scale=2) {

        init_estimate(scale); /// set angles to 0 and U=V=intial_estimate.

        float *r_arr = new float[2*int(R)]; /// R is from 0 to pi, 2*R is from 0 to 2pi.

        int n = (int)N;
        fit_harmonic_series(p_data,r_arr);

        if( log_level > 0 ) {
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
                ctf_ellipse_fit_to_svg(filename,p_frame,(int)N,min_val,max_val,lin_fpix_to_defocus/scale,
                                       c_estimate[k].U,c_estimate[k].V,c_estimate[k].angle,
                                       def_max_angs*SVG_DEF_MARGIN);
            }
        }

        /// To Angstroms
        float def_sign = is_overfocus ? -1.0f : 1.0f;
        for(int k=0;k<K;k++) {
            c_estimate[k].U *= def_sign*lin_fpix_to_defocus/scale;
            c_estimate[k].V *= def_sign*lin_fpix_to_defocus/scale;
            /// The ellipse angle is that of the major axis in the PS frame (x column, y row); calc_def
            /// (create_ctf) puts the largest defocus U at -angle. Written in (-90,90].
            float a = -c_estimate[k].angle*RAD2DEG;
            c_estimate[k].angle = 90.0f - fmodf(fmodf(90.0f - a,180.0f) + 180.0f,180.0f);
        }

        delete [] r_arr;
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
            IO::DefocusIO::write(fp,c_estimate[k],c_prj_wgt[k]);
        }

        fclose(fp);
    }




};

#endif /// CTF_LINEARIZER_H

