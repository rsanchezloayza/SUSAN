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

#ifndef STACK_READER_H
#define STACK_READER_H

#include <pthread.h>
#include <algorithm>
#include "datatypes.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "particles.h"
#include "tomogram.h"
#include "mrc.h"
#include "math_cpu.h"

/// Per-projection statistics of a stack: the alignment fill value and, for the Poisson
/// normalizations, the camera gain a and b = sigma^2 - a mu.
class ProjStatsCalc {

public:
    ProjStatsCalc(int max_k) {
        gat_ok  = new bool[max_k];
        gat_a   = new float[max_k];
        gat_b   = new float[max_k];
        gat_med = new float[max_k];
    }

    ~ProjStatsCalc() {
        delete [] gat_ok;
        delete [] gat_a;
        delete [] gat_b;
        delete [] gat_med;
    }

    void run(ProjStats*p_stats,const float*stack,const int W,const int H,const int K,const int norm_type,const int tomo_id) {
        const bool gat = (norm_type == GAT_RAW) || (norm_type == GAT_NORMAL);
        int n_ok = 0;
        for(int k=0;k<K;k++) {
            const float*img = stack + (size_t)k*W*H;
            p_stats[k].has_fill = Math::detect_fill_border(p_stats[k].fill,img,W,H);
            p_stats[k].gat_a = 1;
            p_stats[k].gat_b = 0;
            if( gat ) {
                gat_ok[k] = Math::estimate_gat_ab(gat_a[k],gat_b[k],img,W,H,p_stats[k].has_fill,p_stats[k].fill,gat_work);
                if( gat_ok[k] ) n_ok++;
            }
        }
        if( !gat )
            return;

        if( n_ok == 0 ) {
            fprintf(stderr," TomoID %d: the camera gain could not be estimated on any projection; using a=1, b=0.\n",tomo_id);
            return;
        }
        float med_a = median_ok(gat_a,K);
        float med_b = median_ok(gat_b,K);
        float min_a=med_a,max_a=med_a,min_b=med_b,max_b=med_b;
        for(int k=0;k<K;k++) {
            if( !gat_ok[k] ) {
                fprintf(stderr," TomoID %d, projection %d: degenerate gain fit, using the median (a=%.3f, b=%.3f).\n",tomo_id,k,med_a,med_b);
                gat_a[k] = med_a;
                gat_b[k] = med_b;
            }
            min_a = fminf(min_a,gat_a[k]); max_a = fmaxf(max_a,gat_a[k]);
            min_b = fminf(min_b,gat_b[k]); max_b = fmaxf(max_b,gat_b[k]);
        }
        printf(" TomoID %d: camera gain a=%.3f [%.3f,%.3f], b=%.3f [%.3f,%.3f] (%d/%d projections fitted).\n",
               tomo_id,med_a,min_a,max_a,med_b,min_b,max_b,n_ok,K);

        if( norm_type == GAT_NORMAL ) {
            for(int k=0;k<K;k++) {
                p_stats[k].gat_a = gat_a[k];
                p_stats[k].gat_b = gat_b[k];
            }
        }
        else if( fabsf(med_a-1) > 0.25f || fabsf(med_b) > 1.0f ) {
            fprintf(stderr," TomoID %d: the stack does not look like raw counts (a=%.3f, b=%.3f); consider poisson_normal.\n",tomo_id,med_a,med_b);
        }
    }

protected:
    float median_ok(const float*p_val,const int K) {
        int n = 0;
        for(int k=0;k<K;k++)
            if( gat_ok[k] ) gat_med[n++] = p_val[k];
        std::nth_element(gat_med,gat_med+n/2,gat_med+n);
        return gat_med[n/2];
    }

    bool  *gat_ok;
    float *gat_a;
    float *gat_b;
    float *gat_med;
    double gat_work[4*GAT_MAX_TILES];

};

class StackReader : public PThread {

public:
    DoubleBufferHandler *double_buffer;
    StackBuffer *buffer_a;
    StackBuffer *buffer_b;
    Tomograms   *tomos;
    Particles   *ptcls;
    uint32      *ptcls_offset;
    uint32      *ptcls_count;
    uint32      *ptcls_tomoid;
    int          norm_type;

    StackReader(Particles*ptcls_in,Tomograms*tomos_in,PBarrier*in_barrier,int in_norm_type=NO_NORM) {
        tomos = tomos_in;
        ptcls = ptcls_in;
        norm_type = in_norm_type;

        long numel = get_tomos_max_numel();

        buffer_a = new StackBuffer(numel,max_k);
        buffer_b = new StackBuffer(numel,max_k);
        stats_calc = new ProjStatsCalc(max_k);

        double_buffer = new DoubleBufferHandler(buffer_a,buffer_b,in_barrier);

        ptcls_count  = new uint32[tomos->num_tomo+1];
        ptcls_offset = new uint32[tomos->num_tomo+1];
        ptcls_tomoid = new uint32[tomos->num_tomo+1];

        memset(ptcls_count,0,sizeof(uint32)*(tomos->num_tomo+1));

        parse_ptcls();
    }

    ~StackReader() {
        delete double_buffer;
        delete buffer_a;
        delete buffer_b;
        delete [] ptcls_count;
        delete [] ptcls_offset;
        delete [] ptcls_tomoid;
        delete stats_calc;
    }

    /// Release both full-stack buffers. Only call after the reader thread has
    /// signalled DONE (i.e. no further stacks will be loaded), otherwise the
    /// producer may still write into them.
    void free_buffers() {
        buffer_a->free_stack();
        buffer_b->free_stack();
    }

protected:
    int max_k;
    ProjStatsCalc *stats_calc;

    /// MAIN LOOP
    void main() {
        int i=0;
        while( ptcls_count[i] > 0 ) {
            StackBuffer*ptr = (StackBuffer*)double_buffer->WO_get_buffer();
            ptr->ptcls.set(ptcls[0],ptcls_offset[i],ptcls_count[i]);
            ptr->tomo_ix = ptcls_tomoid[i];
            load_stack(ptr->stack,tomos->at(ptcls_tomoid[i]));
            Tomogram&tomo = tomos->at(ptcls_tomoid[i]);
            stats_calc->run(ptr->proj_stats,ptr->stack,tomo.stk_dim.x,tomo.stk_dim.y,tomo.stk_dim.z,norm_type,tomo.tomo_id);
            double_buffer->WO_sync(READY);
            i++;
        }
        double_buffer->WO_sync(DONE);
    }

    void load_stack(float*stk_buffer,Tomogram&tomo) {
        Mrc::read(stk_buffer,tomo.stk_dim.x,tomo.stk_dim.y,tomo.stk_dim.z,tomo.stk_name);
    }

    /// CTOR Methods
    long get_tomos_max_numel() {
        uint32 X=0,Y=0,Z=0;
        for(int i=0;i<tomos->num_tomo;i++) {
            if( tomos->at(i).stk_dim.x > X )
                X = tomos->at(i).stk_dim.x;
            if( tomos->at(i).stk_dim.y > Y )
                Y = tomos->at(i).stk_dim.y;
            if( tomos->at(i).stk_dim.z > Z )
                Z = tomos->at(i).stk_dim.z;
        }

        max_k = Z;
        return (long)X*Y*Z;
    }

    /// Index of tomo_id in the tomograms. The particles' tomo_cix field is
    /// deprecated and not read: it is a cached index that goes stale when a
    /// particle set is paired with a different tomostxt.
    uint32 get_tomo_cix(const uint32 tomo_id) {
        int cix = tomos->get_cix(tomo_id);
        if( cix < 0 ) {
            fprintf(stderr,"Particle with tomo_id %d: not found in the tomograms.\n",tomo_id);
            exit(1);
        }
        return (uint32)cix;
    }

    void parse_ptcls() {
        int ptcls_ix = 0;
        int i = 0;
        uint32 cur_tomo_id = 0;
        ptcls_count[ptcls_ix] = 0;
        ptcls_tomoid[ptcls_ix] = 0;
        ptcls_offset[ptcls_ix] = 0;
        Particle cur_ptcl;

        if( ptcls->n_ptcl > ptcls_ix ) {
            ptcls->get(cur_ptcl,i);
            cur_tomo_id = cur_ptcl.tomo_id();
            ptcls_tomoid[ptcls_ix] = get_tomo_cix(cur_tomo_id);
            ptcls_count[ptcls_ix]++;
            i++;

            while( ptcls->get(cur_ptcl,i) ) {
                if( cur_ptcl.tomo_id() != cur_tomo_id ) {
                    ptcls_ix++;
                    if( ptcls_ix >= tomos->num_tomo ) {
                        fprintf(stderr,"Error parsing tomogram info from particles.\n");
                        exit(1);
                    }
                    cur_tomo_id = cur_ptcl.tomo_id();
                    ptcls_tomoid[ptcls_ix] = get_tomo_cix(cur_tomo_id);
                    ptcls_offset[ptcls_ix] = i;
                }
                ptcls_count[ptcls_ix]++;
                i++;
            }
        }
    }

};


#endif /// STACK_READER_H

