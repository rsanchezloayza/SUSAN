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
#include "datatypes.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "particles.h"
#include "tomogram.h"
#include "mrc.h"

class SubstackCrop {

public:
    Tomogram *tomo;
    float x_min;
    float x_max;
    float y_min;
    float y_max;
    int   N;

    SubstackCrop() {
        tomo = NULL;
        N = 200;
    }

    void setup(Tomogram*tomos_in,int box_size) {
        if( tomos_in == NULL ) {
            fprintf(stderr,"SubstackCrop::setup: NULL tomogram pointer.\n");
            exit(1);
        }
        tomo = tomos_in;
        N = box_size;
        float Nh = box_size/2;
        x_min = Nh;
        x_max = float(tomo->stk_dim.x)-Nh;
        y_min = Nh;
        y_max = float(tomo->stk_dim.y)-Nh;
    }

    ~SubstackCrop() {
    }

    bool project_point(V3f&pt_out,const V3f&p_tomo,const int k) {
        if( tomo == NULL )
            return false;
        if( k < 0 || k >= tomo->stk_dim.z )
            return false;
        if( tomo->pix_size == 0 )
            return false;
        pt_out = (tomo->R[k]*p_tomo) + tomo->t[k];
        pt_out = (pt_out/tomo->pix_size) + tomo->stk_center;
        return check_point(pt_out);
    }

    void get_subpix_shift(Vec3&p_subpix,const V3f&p_proj) {
        p_subpix.x = p_proj(0) - floorf(p_proj(0));
        p_subpix.y = p_proj(1) - floorf(p_proj(1));
        p_subpix.z = 0.0f;
    }

    bool check_point(const V3f&p_proj) {
        return ( p_proj(0) >= x_min && p_proj(0) <= x_max && p_proj(1) >= y_min && p_proj(1) <= y_max );
    }

    void crop(float*substack,float*stack,const V3f&p_proj,const int k) {
        internal_crop( substack+k*N*N, stack+k*tomo->stk_dim.x*tomo->stk_dim.y, (int)floorf(p_proj(0)), (int)floorf(p_proj(1)) );
    }

    float normalize_zero_mean(float*substack,const int k) {
        return internal_normalize_zero_mean( substack+k*N*N );
    }

    void normalize_zero_mean_one_std(float*substack,const int k) {
        internal_normalize_zero_mean_new_std( substack+k*N*N, 1.0f );
    }

    void normalize_zero_mean_w_std(float*substack,float w,const int k) {
        internal_normalize_zero_mean_new_std( substack+k*N*N, w );
    }


protected:

    void internal_crop(float*substack,float*stack,const int x,const int y) {
        int w = tomo->stk_dim.x;
        float *out = substack;
        float *in = stack + x-(N/2) + (y-(N/2))*w;
        for(int j=0;j<N;j++) {
            memcpy((void*)out,(void*)in,N*sizeof(float));
            out += N;
            in  += w;
        }
    }

    float internal_normalize_zero_mean(float*substack) {
        float avg,std;
        Math::get_avg_std(avg,std,substack,N*N);
        Math::zero_mean(substack,N*N,avg);
        return std;
    }

    void internal_normalize_zero_mean_new_std(float*substack,float new_std) {
        float avg,std;
        Math::get_avg_std(avg,std,substack,N*N);
        Math::normalize(substack,N*N,avg,std,new_std);
    }

};


#endif /// SUBSTACK_CROP_H

