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

#ifndef POOL_COORDINATOR_H
#define POOL_COORDINATOR_H

#include <pthread.h>
#include "datatypes.h"
#include "memory.h"
#include "thread_sharing.h"
#include "thread_base.h"
#include "particles.h"
#include "tomogram.h"
#include "stack_reader.h"

class PoolCoordinator : public PThread {

public:
    virtual ~PoolCoordinator() = default;

public:
    StackReader &stkrdr;
    int num_threads;

    PoolCoordinator(StackReader&in_stkrdr,int in_num_threads)
        : stkrdr(in_stkrdr), num_threads(in_num_threads)
    {
    }

protected:
    void main() {
        coord_init();
        Status_t s = stkrdr.double_buffer->RO_get_status();
        while( s > DONE ) {
            if( s == READY ) {
                StackBuffer*ptr = (StackBuffer*)stkrdr.double_buffer->RO_get_buffer();
                coord_main(ptr->stack,ptr->ptcls,stkrdr.tomos->at(ptr->tomo_ix));
            }
            stkrdr.double_buffer->RO_sync();
            s = stkrdr.double_buffer->RO_get_status();
        }
        coord_end();
    }

    virtual void coord_init() {
    }

    virtual void coord_main(float*stack,ParticlesSubset&ptcls,Tomogram&tomo) {
        printf(" Processing TomoID %d: %d particles\n",tomo.tomo_id,ptcls.n_ptcl);
        if( ptcls.n_ptcl > 0 ) {
            Particle ptcl;
            ptcls.get(ptcl,0);
            printf("   First Particle ID: %d [%d]\n",ptcl.ptcl_id(),ptcl.tomo_id());
            ptcls.get(ptcl,ptcls.n_ptcl-1);
            printf("   Last Particle ID: %d [%d]\n",ptcl.ptcl_id(),ptcl.tomo_id());
        }
    }

    virtual void coord_end() {
    }

};


#endif /// POOL_COORDINATOR_H

