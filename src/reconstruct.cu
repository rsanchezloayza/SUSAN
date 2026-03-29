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
#include "reconstruct.h"
#include "particles.h"
#include "tomogram.h"
#include "reconstruct_args.h"
#include "datatypes.h"

void print_data_info(Particles&ptcls,Tomograms&tomos,ArgsRec::Info&info) {
    if(info.verbosity==VERBOSITY_FULL) {
        printf("\t\tAvailable particles:  %d.\n",ptcls.n_ptcl);
        printf("\t\tNumber of classes:    %d.\n",ptcls.n_refs);
    	printf("\t\tTomograms available:  %d.\n",tomos.num_tomo);
    	printf("\t\tAvailable projections: %d (max).\n",tomos.num_proj);
    }
    else {
        printf("    - %d Particles (%d classes) in %d tomograms with max %d projections.\n",ptcls.n_ptcl,ptcls.n_refs,tomos.num_tomo,tomos.num_proj);
    }
}

int main(int ac, char** av) {

    ArgsRec::Info info;

    if( ArgsRec::parse_args(info,ac,av) ) {

        ArgsRec::print(info);

        PBarrier barrier(2);

        if( info.verbosity != VERBOSITY_MINIMAL ) {
            printf("\tLoading data files..."); fflush(stdout);}
        ParticlesRW ptcls(info.ptcls_in);
        Tomograms tomos(info.tomos_in);
        if( info.verbosity != VERBOSITY_MINIMAL ) {
            printf(" Done\n"); fflush(stdout);
        }

        print_data_info(ptcls,tomos,info);

        StackReader stkrdr(&ptcls,&tomos,&barrier);
        RecPool pool(&info,ptcls.n_refs,tomos.num_proj,ptcls.n_ptcl,stkrdr,info.n_threads);

        stkrdr.start();
        pool.start();

        stkrdr.wait();
        pool.wait();
    }
    else {
        fprintf(stderr,"Error parsing input arguments.\n");
        exit(1);
    }

    return 0;
}



