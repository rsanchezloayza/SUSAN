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

#ifndef SUBSTACK_CROP_H
#define SUBSTACK_CROP_H

#include <pthread.h>
#include <random>
#include <vector>
#include <algorithm>
#include "datatypes.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "particles.h"
#include "tomogram.h"
#include "mrc.h"

/// Crop that may leave the frame: a projection is used when the box centre is inside the frame
/// and not on the fill value. Normalization over the valid pixels (eroded by EDGE_ERODE); the
/// invalid ones are filled as the padding: noise with the valid mean / std, or 0.
class SubstackCrop {

public:
    Tomogram        *tomo;
    const ProjStats *stats;
    int N;
    int norm_type;
    int pad_type;

    SubstackCrop() {
        tomo  = NULL;
        stats = NULL;
        N = 0;
        norm_type = NO_NORM;
        pad_type  = PAD_ZERO;
    }

    void init(int box_size,int in_norm_type,int in_pad_type) {
        N = box_size;
        norm_type = in_norm_type;
        pad_type  = in_pad_type;
        valid.resize((size_t)N*N);
        tmp.resize((size_t)N*N);
        if( norm_type == LOCAL_NORM ) {
            sat_s .resize((size_t)(N+1)*(N+1));
            sat_s2.resize((size_t)(N+1)*(N+1));
            sat_n .resize((size_t)(N+1)*(N+1));
            win_px.assign(1,std::max(1,(int)lroundf(SUSAN_LCN_NOREF_FRAC*N)));
        }
    }

    /// Reference radii (A) for the local normalization, one per reference.
    void set_local_radii(const std::vector<float>&R_angstrom) {
        radii_A = R_angstrom;
    }

    void setup(Tomogram*tomo_in,const ProjStats*stats_in) {
        tomo  = tomo_in;
        stats = stats_in;
        if( norm_type == LOCAL_NORM && !radii_A.empty() && tomo->pix_size > 0 ) {
            win_px.resize(radii_A.size()+1);
            int largest = 1;
            for(size_t r=0;r<radii_A.size();r++) {
                win_px[r] = std::max(1,(int)lroundf(SUSAN_LCN_MARGIN*radii_A[r]/tomo->pix_size));
                largest = std::max(largest,win_px[r]);
            }
            win_px[radii_A.size()] = largest;
        }
    }

    /// ref_ix < 0: window of the largest reference (or the box fraction without references).
    bool crop(float*ss,float2&pad,const float*stack,const V3f&p_proj,const int k,const uint32 seed_id,const int ref_ix=-1) {
        const int W = tomo->stk_dim.x;
        const int H = tomo->stk_dim.y;
        const int cx = (int)floorf(p_proj(0));
        const int cy = (int)floorf(p_proj(1));
        pad.x = 0;
        pad.y = 1;
        if( cx < 0 || cy < 0 || cx >= W || cy >= H )
            return false;

        const ProjStats&ps = stats[k];
        const float*img = stack + (size_t)k*W*H;
        const int x0 = cx - N/2;
        const int y0 = cy - N/2;
        const int xs = std::max(x0,0)   - x0;
        const int xe = std::min(x0+N,W) - x0;
        int n_bad = 0;
        for(int j=0;j<N;j++) {
            float*out = ss + j*N;
            unsigned char*vld = valid.data() + j*N;
            const int Y = y0 + j;
            if( Y < 0 || Y >= H ) {
                memset(out,0,sizeof(float)*N);
                memset(vld,0,N);
                n_bad += N;
                continue;
            }
            for(int i=0;i<xs;i++) { out[i] = 0; vld[i] = 0; }
            for(int i=xe;i<N;i++) { out[i] = 0; vld[i] = 0; }
            n_bad += N - (xe-xs);
            memcpy(out+xs,img+(size_t)Y*W+x0+xs,sizeof(float)*(xe-xs));
            memset(vld+xs,1,xe-xs);
            if( ps.has_fill ) {
                for(int i=xs;i<xe;i++) {
                    if( Math::is_fill(out[i],ps.fill) ) { vld[i] = 0; n_bad++; }
                }
            }
        }

        if( n_bad > 0 ) {
            erode();
            if( !valid[N/2 + (N/2)*N] )
                return false;
        }

        if( norm_type == GAT_RAW || norm_type == GAT_NORMAL )
            Math::gat_transform(ss,N*N,ps.gat_a,ps.gat_b);

        float avg,std;
        if( n_bad == 0 ) {
            Math::get_avg_std(avg,std,ss,N*N);
        }
        else {
            double s=0,s2=0;
            int count=0;
            for(int i=0;i<N*N;i++) {
                if( valid[i] ) { s += ss[i]; s2 += (double)ss[i]*ss[i]; count++; }
            }
            if( count < EDGE_MIN_VALID )
                return false;
            avg = (float)(s/count);
            std = (float)sqrt( fmax(s2/count - (s/count)*(s/count),0.0) );
        }
        if( std < SUSAN_FLOAT_TOL || std::isnan(std) || std::isinf(std) )
            return false;

        if( norm_type == LOCAL_NORM ) {
            int Rw = win_px.back();
            if( ref_ix >= 0 && ref_ix + 1 < (int)win_px.size() )
                Rw = win_px[ref_ix];
            local_normalize(ss,Rw);
            pad.x = 0;
            pad.y = 1;
        }
        else if( norm_type == NO_NORM ) {
            pad.x = avg;
            pad.y = std;
        }
        else if( norm_type == ZERO_MEAN_1_STD ) {
            Math::normalize(ss,N*N,avg,std);
        }
        else {
            Math::normalize(ss,N*N,avg,1.0);
            pad.y = std;
        }

        if( n_bad > 0 ) {
            if( pad_type == PAD_GAUSSIAN ) {
                rng.seed( seed_id*1000003u + (uint32)k );
                nrm.reset();
                for(int i=0;i<N*N;i++)
                    if( !valid[i] ) ss[i] = pad.x + pad.y*nrm(rng);
            }
            else {
                for(int i=0;i<N*N;i++)
                    if( !valid[i] ) ss[i] = 0;
            }
        }
        return true;
    }

protected:
    std::vector<unsigned char> valid;
    std::vector<unsigned char> tmp;
    std::vector<double> sat_s;
    std::vector<double> sat_s2;
    std::vector<int>    sat_n;
    std::vector<float>  radii_A;
    std::vector<int>    win_px;
    std::mt19937 rng;
    std::normal_distribution<float> nrm{0.0f,1.0f};

    /// Per-pixel (x-mean)/std over the valid pixels of the (2Rw+1)^2 window (summed-area tables).
    void local_normalize(float*ss,const int Rw) {
        const int S = N+1;
        for(int i=0;i<S;i++) { sat_s[i] = 0; sat_s2[i] = 0; sat_n[i] = 0; }
        for(int j=0;j<N;j++) {
            const size_t r0 = (size_t)j*S;
            const size_t r1 = (size_t)(j+1)*S;
            sat_s[r1] = 0; sat_s2[r1] = 0; sat_n[r1] = 0;
            double rs=0,rs2=0;
            int rn=0;
            for(int i=0;i<N;i++) {
                if( valid[j*N+i] ) {
                    const double v = ss[j*N+i];
                    rs += v; rs2 += v*v; rn++;
                }
                sat_s [r1+i+1] = sat_s [r0+i+1] + rs;
                sat_s2[r1+i+1] = sat_s2[r0+i+1] + rs2;
                sat_n [r1+i+1] = sat_n [r0+i+1] + rn;
            }
        }
        for(int j=0;j<N;j++) {
            const int ya = std::max(j-Rw,0);
            const int yb = std::min(j+Rw+1,N);
            for(int i=0;i<N;i++) {
                const int xa = std::max(i-Rw,0);
                const int xb = std::min(i+Rw+1,N);
                const size_t a = (size_t)ya*S+xa, b = (size_t)ya*S+xb, c = (size_t)yb*S+xa, d = (size_t)yb*S+xb;
                const int    n  = sat_n [d] - sat_n [b] - sat_n [c] + sat_n [a];
                const double s1 = sat_s [d] - sat_s [b] - sat_s [c] + sat_s [a];
                const double s2 = sat_s2[d] - sat_s2[b] - sat_s2[c] + sat_s2[a];
                float val = 0;
                if( n >= 2 ) {
                    const double mu = s1/n;
                    const double sd = sqrt( fmax(s2/n - mu*mu,0.0) );
                    if( sd >= SUSAN_FLOAT_TOL )
                        val = (float)( (ss[j*N+i]-mu)/sd );
                }
                ss[j*N+i] = val;
            }
        }
    }

    void erode() {
        for(int it=0;it<EDGE_ERODE;it++) {
            memcpy(tmp.data(),valid.data(),(size_t)N*N);
            for(int j=0;j<N;j++) {
                for(int i=0;i<N;i++) {
                    int ix = i + j*N;
                    if( !tmp[ix] ) continue;
                    if( (i>0 && !tmp[ix-1]) || (i<N-1 && !tmp[ix+1]) || (j>0 && !tmp[ix-N]) || (j<N-1 && !tmp[ix+N]) )
                        valid[ix] = 0;
                }
            }
        }
    }

};

#endif /// SUBSTACK_CROP_H

