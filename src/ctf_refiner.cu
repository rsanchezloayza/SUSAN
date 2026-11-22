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
#include "tomogram.h"
#include "particles.h"
#include "reference.h"
#include "ctf_refiner.h"
#include "ctf_refiner_args.h"


int main(int ac, char** av) {

    ArgsCtfRef::Info info;

    if( ArgsCtfRef::parse_args(info,ac,av) ) {
        ArgsCtfRef::print(info);
        PBarrier barrier(2);
        DataInfo::print_loading(info.verbosity);
        ParticlesRW ptcls(info.ptcls_in);
        References refs(info.refs_file);
        Tomograms tomos(info.tomo_file);
        DataInfo::print_loaded(info.verbosity);
        DataInfo::print_data_info(ptcls,tomos,info.verbosity);
        StackReader stkrdr(&ptcls,&tomos,&barrier,info.norm_type);
        CtfRefinerPool pool(&info,&refs,tomos.num_proj,ptcls.n_ptcl,stkrdr,info.n_threads);

        /// output_type refinement_stats: pooled CTF statistics (ctf_stats.h)
        CtfStats stats;
        const bool save_stats = info.output_type == CTF_OUTPUT_REFINEMENT_STATS;
        if( save_stats ) {
            stats.setup(tomos,std::vector<int>(info.stats_groups,info.stats_groups+info.n_stats_groups),info.n_threads,
                        info.use_halves,info.box_size+info.pad_size,info.fpix_max,info.box_size,info.def_range,info.def_step);
            stats.set_layout(ptcls,tomos);
            stats.set_range(0,ptcls.n_ptcl,0,ptcls.n_ptcl);
            pool.set_stats(&stats,ptcls.p_raw);
        }

        /// output_type refinement_covar: per projection evidence at the solution (ali_stats.h)
        AliStats covar;
        const bool save_covar = info.output_type == CTF_OUTPUT_REFINEMENT_COVAR;
        if( save_covar ) {
            float steps[AliCovar::NC_CTF-1];
            ArgsCtfRef::covar_steps(info,steps);
            covar.setup(tomos,CIRCLE,info.off_x,info.off_y,info.off_z,info.off_s,std::vector<int>(),
                        info.n_threads,info.use_halves,std::vector<M33f>(1,M33f::Identity()));
            covar.setup_covar(AliCovar::NC_CTF,AliCovar::FIELDS_CTF,steps);
            covar.set_layout(ptcls,tomos);
            covar.set_range(0,ptcls.n_ptcl,0,ptcls.n_ptcl);
            pool.set_covar(&covar,ptcls.p_raw);
        }

        stkrdr.start();
        pool.start();

        stkrdr.wait();
        pool.wait();

        ptcls.save(info.ptcls_out);

        if( save_stats && !stats.save(info.ptcls_out) )
            fprintf(stderr,"Error writing the statistics of %s\n",info.ptcls_out);
        if( save_covar && !covar.save(info.ptcls_out) )
            fprintf(stderr,"Error writing the covariance evidence of %s\n",info.ptcls_out);
    }
    else {
        DataInfo::exit_bad_args();
    }
	
    return 0;
}



