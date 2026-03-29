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

#ifndef MPI_IFACE_H
#define MPI_IFACE_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <mpi.h>

#include "datatypes.h"
#include "particles.h"

class MpiScatterInfo {
public:
    uint32 *count_per_node;
    uint32 num_nodes;

    MpiScatterInfo(uint32 in_num_nodes) {
        num_nodes = in_num_nodes;
        count_per_node = new uint32[num_nodes];
        for(uint32 i=0;i<num_nodes;i++) {
            count_per_node[i] = 0;
        }
    }

    ~MpiScatterInfo() {
        delete [] count_per_node;
    }

    void simple_scatter(uint32 numel) {
        for(uint32 i=0;i<numel;i++) {
            count_per_node[ i%num_nodes ]++;
        }
    }
};

class MpiProgress {
protected:
    unsigned int*shared_buffer;
    MPI_Win win;

public:
    MpiProgress(int num_nodes) {
        MPI_Win_allocate(num_nodes*sizeof(unsigned int), sizeof(unsigned int), MPI_INFO_NULL, MPI_COMM_WORLD, &shared_buffer, &win);
    }

    void put(int ix, unsigned int value) {
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, ix, 0, win);
        MPI_Put(&value,1,MPI_UNSIGNED,ix,ix,1,MPI_UNSIGNED,win);
        MPI_Win_unlock(ix, win);
    }

    unsigned int get(int ix) {
        unsigned int value;
        MPI_Win_lock(MPI_LOCK_SHARED, ix, 0, win);
        MPI_Get(&value,1,MPI_UNSIGNED,ix,ix,1,MPI_UNSIGNED,win);
        MPI_Win_unlock(ix, win);
        return value;
    }

    ~MpiProgress() {
        MPI_Win_free(&win);
    }
};

class MpiInterface {

public:
    int num_nodes;
    int node_id;

    MpiInterface() : num_nodes(0), node_id(0)
    {
        int err = MPI_Init(NULL,NULL);
        if( err != MPI_SUCCESS ) {
            fprintf(stderr,"Error initializing MPI.\n");
            exit(1);
        }
        MPI_Comm_size(MPI_COMM_WORLD,&num_nodes);
        MPI_Comm_rank(MPI_COMM_WORLD,&node_id);
    }

    ~MpiInterface() {
        MPI_Finalize();
    }

    bool is_main_node() {
        return node_id==0;
    }

    bool is_multi_node() {
        return num_nodes>1;
    }

    void print_info(int verbosity=1) {
        if( verbosity > 0 )
            printf("\t\tMPI nodes: %d\n",num_nodes);
        else
            printf("    - MPI nodes: %d\n",num_nodes);
    }

    Particles*scatter_particles(MpiScatterInfo&scatter_info,Particles*ptcls_in) {
        uint32 data[3];
        if( is_main_node() ) {

            scatter_info.simple_scatter(ptcls_in->n_ptcl);

            data[1] = ptcls_in->n_proj;
            data[2] = ptcls_in->n_refs;

            for(int i=1;i<num_nodes;i++) {
                data[0] = scatter_info.count_per_node[i];
                MPI_Send((void*)data,sizeof(uint32)*3,MPI_BYTE,i,0,MPI_COMM_WORLD);
            }

            int offset = scatter_info.count_per_node[0];
            ParticlesSubset ptcls_scatter;
            std::vector<MPI_Request> reqs(num_nodes-1);
            for(int i=1;i<num_nodes;i++) {
                ptcls_scatter.set((*ptcls_in),offset,scatter_info.count_per_node[i]);
                MPI_Isend((void*)ptcls_scatter.p_raw,ptcls_scatter.n_bytes*ptcls_scatter.n_ptcl,MPI_BYTE,i,1,MPI_COMM_WORLD,&reqs[i-1]);
                offset += scatter_info.count_per_node[i];
            }
            MPI_Waitall(num_nodes-1,reqs.data(),MPI_STATUSES_IGNORE);

            ParticlesSubset*ptcls_rslt = new ParticlesSubset;
            ptcls_rslt->set((*ptcls_in),0,scatter_info.count_per_node[0]);
            return ptcls_rslt;
        }
        else {
            MPI_Recv((void*)data,sizeof(uint32)*3,MPI_BYTE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

            ParticlesMem*ptcls_rslt = new ParticlesMem(data[0],data[1],data[2]);
            MPI_Recv((void*)ptcls_rslt->p_raw,ptcls_rslt->n_bytes*ptcls_rslt->n_ptcl,MPI_BYTE,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            return ptcls_rslt;
        }
    }

    void gather_particles(Particles*ptcls_scattered,MpiScatterInfo&scatter_info,Particles*ptcls_in) {
        if( is_main_node() ) {

            int offset = scatter_info.count_per_node[0];
            ParticlesSubset ptcls_gather;
            std::vector<MPI_Request> reqs(num_nodes-1);
            for(int i=1;i<num_nodes;i++) {
                ptcls_gather.set((*ptcls_in),offset,scatter_info.count_per_node[i]);
                MPI_Irecv((void*)ptcls_gather.p_raw,ptcls_gather.n_bytes*ptcls_gather.n_ptcl,MPI_BYTE,i,2,MPI_COMM_WORLD,&reqs[i-1]);
                offset += scatter_info.count_per_node[i];
            }
            MPI_Waitall(num_nodes-1,reqs.data(),MPI_STATUSES_IGNORE);
        }
        else {
            MPI_Send((void*)ptcls_scattered->p_raw,ptcls_scattered->n_bytes*ptcls_scattered->n_ptcl,MPI_BYTE,0,2,MPI_COMM_WORLD);
        }
    }



    void delete_scattered_ptcls(Particles*ptcls_in) {
        delete ptcls_in;
    }
};

#endif /// MPI_IFACE_H
