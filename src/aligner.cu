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
#include "datatypes.h"


int main(int ac, char** av) {

    ArgsAli::Info info;

    if( ArgsAli::parse_args(info,ac,av) ) {
        ArgsAli::print(info);
        PBarrier barrier(2);
        DataInfo::print_loading(info.verbosity);
        ParticlesRW ptcls(info.ptcls_in);
        References refs(info.refs_file);
        Tomograms tomos(info.tomo_file);
        DataInfo::print_loaded(info.verbosity);
        DataInfo::print_data_info(ptcls,tomos,info.verbosity);
        if( info.output_type == ALI_OUTPUT_DENSE && info.type == 3 )
            DenseIO::write_manifest(info,ptcls,tomos);
        StackReader stkrdr(&ptcls,&tomos,&barrier,info.norm_type);
        AliPool pool(&info,&refs,tomos.num_proj,ptcls.n_ptcl,stkrdr,info.n_threads);

        /// type 2: per projection and grouped statistics of the CC evidence (ali_stats.h)
        AliStats stats;
        const bool save_stats = info.output_type == ALI_OUTPUT_ALIGNMENT_STATS;
        if( save_stats ) {
            stats.setup(tomos,info.off_type,info.off_x,info.off_y,info.off_z,info.off_s,
                        std::vector<int>(info.stats_groups,info.stats_groups+info.n_stats_groups),info.n_threads,
                        info.ali_halves,ArgsAli::stats_angles(info));
            stats.set_layout(ptcls,tomos);
            stats.set_range(0,ptcls.n_ptcl,0,ptcls.n_ptcl);
            pool.set_stats(&stats,ptcls.p_raw);
        }

        stkrdr.start();
        pool.start();

        stkrdr.wait();
        pool.wait();

        if( info.output_type != ALI_OUTPUT_DENSE )
            ptcls.save(info.output);

        if( save_stats && !stats.save(info.output) )
            fprintf(stderr,"Error writing the statistics of %s\n",info.output);
    }
    else {
        DataInfo::exit_bad_args();
    }

    return 0;
}



