/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <unistd.h>

#include "io.h"
#include "data_info.h"
#include "aligner.h"
#include "particles.h"
#include "tomogram.h"
#include "reference.h"
#include "aligner_args.h"
#include "mpi_iface.h"
#include "progress.h"

class AliPoolMpi : public AliPool {

public:
    MpiInterface*mpi_iface;
    MpiProgress *mpi_progress;

    AliPoolMpi(ArgsAli::Info*info,References*in_p_refs,int in_max_K,int num_ptcls,StackReader&stkrdr,int in_num_threads)
        : AliPool(info,in_p_refs,in_max_K,num_ptcls,stkrdr,in_num_threads)
    {
    }

    void set_mpi(MpiInterface*in_mpi_iface,MpiProgress*in_mpi_progress,Particles*ptcls_full) {
        mpi_iface = in_mpi_iface;
        mpi_progress = in_mpi_progress;
        unsigned int val = 0;
        mpi_progress->put( mpi_iface->node_id, val );

        if( mpi_iface->is_main_node() ) {
            progress.set_total( ptcls_full->n_ptcl );
        }
    }

protected:
    unsigned int count_progress_mpi() {
        unsigned int rslt = 0;
        if( mpi_iface->is_main_node() ) {
            for(unsigned int i=0;i<mpi_iface->num_nodes;i++) {
                rslt += mpi_progress->get(i);
            }
        }
        return rslt;
    }

    void progress_start() {
        if( mpi_iface->is_main_node() ) {
            AliPool::progress_start();
        }
    }

    void show_progress(const int ptcls_in_tomo) {

        while( (count_progress()) < ptcls_in_tomo ) {
            unsigned int local_progress = count_accumul();
            mpi_progress->put( mpi_iface->node_id, local_progress );

            if( mpi_iface->is_main_node() ) {
                unsigned int total_progress = count_progress_mpi();
                progress.update(total_progress,false);
            }
            sleep(3);
        } /// while
    }

    void show_done() {
        if( mpi_iface->is_main_node() ) {
            progress.finish();
        }
    }
};


int main(int ac, char** av) {

    MpiInterface mpi_iface;
    ArgsAli::Info info;

    if( ArgsAli::parse_args(info,ac,av) ) {

        if( mpi_iface.is_main_node() ) {
            ArgsAli::print(info);
            mpi_iface.print_info(info.verbosity);
        }

        PBarrier barrier(2);
        ParticlesRW*ptcls_io;
        MpiScatterInfo scatter_info(mpi_iface.num_nodes);
        MpiProgress mpi_progress(mpi_iface.num_nodes);

        if( mpi_iface.is_main_node() ) {
            printf("\tLoading and scattering data files...");
            fflush(stdout);
        }

        References refs(info.refs_file);
        Tomograms tomos(info.tomo_file);

        if( mpi_iface.is_main_node() ) {
            ptcls_io = new ParticlesRW(info.ptcls_in);
            if( info.output_type == ALI_OUTPUT_DENSE && info.type == 3 )
                DenseIO::write_manifest(info,*ptcls_io,tomos);
        }

        Particles*ptcls = mpi_iface.scatter_particles(scatter_info,ptcls_io);

        if( mpi_iface.is_main_node() ) {
            printf(" Done\n");
            fflush(stdout);
        }

        if( mpi_iface.is_main_node() ) {
            DataInfo::print_data_info(ptcls_io,tomos,info.verbosity);
        }

        StackReader stkrdr(ptcls,&tomos,&barrier,info.norm_type);
        AliPoolMpi pool(&info,&refs,tomos.num_proj,ptcls->n_ptcl,stkrdr,info.n_threads);
        pool.set_mpi(&mpi_iface,&mpi_progress,ptcls_io);

        /// type 2 statistics (ali_stats.h): the main node computes the layout (tomogram and group labels of every
        /// particle) and holds the rows of all the particles; each node processes its contiguous range.
        AliStats stats;
        const bool save_stats = info.output_type == ALI_OUTPUT_ALIGNMENT_STATS;
        if( save_stats ) {
            stats.setup(tomos,info.off_type,info.off_x,info.off_y,info.off_z,info.off_s,
                        std::vector<int>(info.stats_groups,info.stats_groups+info.n_stats_groups),info.n_threads,
                        info.ali_halves,ArgsAli::stats_angles(info));
            mpi_iface.bcast_scatter(scatter_info);
            std::vector<uint8_t> layout;
            if( mpi_iface.is_main_node() ) {
                stats.set_layout(*ptcls_io,tomos);
                layout = stats.pack_layout();
            }
            mpi_iface.bcast_bytes(layout);
            if( !mpi_iface.is_main_node() )
                stats.unpack_layout(layout);
            int lo = mpi_iface.scatter_offset(scatter_info);
            int hi = lo + scatter_info.count_per_node[mpi_iface.node_id];
            if( mpi_iface.is_main_node() )
                stats.set_range(lo,hi,0,stats.n_ptcl);
            else
                stats.set_range(lo,hi,lo,hi-lo);
            pool.set_stats(&stats,ptcls->p_raw);
        }

        stkrdr.start();
        pool.start();

        stkrdr.wait();
        pool.wait();

        if( info.output_type != ALI_OUTPUT_DENSE ) {
            if( mpi_iface.is_main_node() ) {
                printf("\tGathering data files...");
                fflush(stdout);
            }

            mpi_iface.gather_particles(ptcls,scatter_info,ptcls_io);

            if( mpi_iface.is_main_node() ) {
                ptcls_io->save(info.output);
                printf(" Done\n");
                fflush(stdout);
            }
        }

        if( save_stats ) {
            if( mpi_iface.is_main_node() ) {
                std::vector<uint8_t> b;
                for(int i=1;i<mpi_iface.num_nodes;i++) {
                    mpi_iface.recv_bytes(i,b,3);
                    stats.merge_node(b);
                }
                stats.finish();
                if( !stats.save(info.output) )
                    fprintf(stderr,"Error writing the statistics of %s\n",info.output);
            }
            else {
                mpi_iface.send_bytes(0,stats.pack_node(),3);
            }
        }

        mpi_iface.delete_scattered_ptcls(ptcls);

        if( mpi_iface.is_main_node() ) {
            delete ptcls_io;
        }
    }
    else {
        DataInfo::exit_bad_args();
    }
	
    return 0;
}



