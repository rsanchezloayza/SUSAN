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
#include "datatypes.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "particles.h"
#include "tomogram.h"
#include "mrc.h"

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

    StackReader(Particles*ptcls_in,Tomograms*tomos_in,PBarrier*in_barrier) {
        tomos = tomos_in;
        ptcls = ptcls_in;

        long numel = get_tomos_max_numel();

        buffer_a = new StackBuffer(numel);
        buffer_b = new StackBuffer(numel);

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
    }

    /// Release both full-stack buffers. Only call after the reader thread has
    /// signalled DONE (i.e. no further stacks will be loaded), otherwise the
    /// producer may still write into them.
    void free_buffers() {
        buffer_a->free_stack();
        buffer_b->free_stack();
    }

protected:

    /// MAIN LOOP
    void main() {
        int i=0;
        while( ptcls_count[i] > 0 ) {
            StackBuffer*ptr = (StackBuffer*)double_buffer->WO_get_buffer();
            ptr->ptcls.set(ptcls[0],ptcls_offset[i],ptcls_count[i]);
            ptr->tomo_ix = ptcls_tomoid[i];
            load_stack(ptr->stack,tomos->at(ptcls_tomoid[i]));
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

        return X*Y*Z;
    }

    void parse_ptcls() {
        int ptcls_ix = 0;
        int i = 0;
        ptcls_count[ptcls_ix] = 0;
        ptcls_tomoid[ptcls_ix] = 0;
        ptcls_offset[ptcls_ix] = 0;
        Particle cur_ptcl;

        if( ptcls->n_ptcl > ptcls_ix ) {
            ptcls->get(cur_ptcl,i);
            ptcls_tomoid[ptcls_ix] = cur_ptcl.tomo_cix();
            ptcls_count[ptcls_ix]++;
            i++;

            while( ptcls->get(cur_ptcl,i) ) {
                if( cur_ptcl.tomo_cix() != ptcls_tomoid[ptcls_ix] ) {
                    ptcls_ix++;
                    if( ptcls_ix >= tomos->num_tomo ) {
                        fprintf(stderr,"Error parsing tomogram info from particles.\n");
                        exit(1);
                    }
                    ptcls_tomoid[ptcls_ix] = cur_ptcl.tomo_cix();
                    ptcls_offset[ptcls_ix] = i;
                }
                ptcls_count[ptcls_ix]++;
                i++;
            }
        }
    }

};


#endif /// STACK_READER_H

