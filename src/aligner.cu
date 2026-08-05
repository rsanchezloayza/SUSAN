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
        StackReader stkrdr(&ptcls,&tomos,&barrier);
        AliPool pool(&info,&refs,tomos.num_proj,ptcls.n_ptcl,stkrdr,info.n_threads);

        stkrdr.start();
        pool.start();

        stkrdr.wait();
        pool.wait();

        ptcls.save(info.ptcls_out);
    }
    else {
        DataInfo::exit_bad_args();
    }

    return 0;
}



