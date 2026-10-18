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
    }

    void setup(Tomogram*tomo_in,const ProjStats*stats_in) {
        tomo  = tomo_in;
        stats = stats_in;
    }

    bool crop(float*ss,float2&pad,const float*stack,const V3f&p_proj,const int k,const uint32 seed_id) {
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

        if( norm_type == NO_NORM ) {
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
    std::mt19937 rng;
    std::normal_distribution<float> nrm{0.0f,1.0f};

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

