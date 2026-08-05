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
#include "crop_projections.h"
#include "particles.h"
#include "tomogram.h"
#include "crop_projections_args.h"


int main(int ac, char** av) {

    ArgsCropProjections::Info info;

    if( ArgsCropProjections::parse_args(info,ac,av) ) {
        ArgsCropProjections::print(info);
        PBarrier barrier(2);
        DataInfo::print_loading(VERBOSITY_BASIC);
        ParticlesRW ptcls(info.ptcls_in);
        Tomograms tomos(info.tomos_in);
        DataInfo::print_loaded(VERBOSITY_BASIC);
        DataInfo::print_data_info(ptcls,tomos,VERBOSITY_BASIC);
        StackReader stkrdr(&ptcls,&tomos,&barrier);
        CropProjectionsPool pool(&info,tomos.num_proj,ptcls.n_ptcl,stkrdr,info.n_threads);

        stkrdr.start();
        pool.start();

        stkrdr.wait();
        pool.wait();
    }
    else {
        DataInfo::exit_bad_args();
    }

    return 0;
}



