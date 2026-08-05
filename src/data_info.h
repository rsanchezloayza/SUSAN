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

#ifndef DATA_INFO_H
#define DATA_INFO_H

#include <cstdio>
#include <vector>

#include "datatypes.h"
#include "particles.h"
#include "tomogram.h"

namespace DataInfo {

/// Tomograms referenced by the particles. n_orphan counts particles whose
/// tomo_id is missing from the tomograms file.
inline int count_used_tomos(Particles&ptcls,Tomograms&tomos,int&n_orphan) {

    std::vector<bool> used(tomos.num_tomo,false);
    Particle ptcl;

    uint32 last_id = 0;
    int    last_ix = -1;

    n_orphan = 0;

    for(uint32 i=0;i<ptcls.n_ptcl;i++) {
        ptcls.get(ptcl,i);
        uint32 tid = ptcl.tomo_id();
        /// Particles are grouped by tomogram, so the previous hit is almost
        /// always the answer and get_cix's linear scan is skipped.
        if( last_ix < 0 || tid != last_id ) {
            last_id = tid;
            last_ix = tomos.get_cix(tid);
        }
        if( last_ix >= 0 ) used[last_ix] = true;
        else               n_orphan++;
    }

    int n_used = 0;
    for(uint32 i=0;i<tomos.num_tomo;i++)
        if( used[i] ) n_used++;

    return n_used;
}

inline void format_tomo_count(char*buffer,int n_used,uint32 n_avail) {
    if( n_used < (int)n_avail ) sprintf(buffer,"%d/%d",n_used,n_avail);
    else                        sprintf(buffer,"%d",n_avail);
}

inline void print_data_info(Particles&ptcls,Tomograms&tomos,int verbosity) {

    int  n_orphan;
    int  n_used = count_used_tomos(ptcls,tomos,n_orphan);
    char tomo_str[32];
    format_tomo_count(tomo_str,n_used,tomos.num_tomo);

    if( verbosity == VERBOSITY_FULL ) {
        printf("\t\tAvailable particles:   %d.\n",ptcls.n_ptcl);
        printf("\t\tNumber of classes:     %d.\n",ptcls.n_refs);
        printf("\t\tTomograms used:        %s.\n",tomo_str);
        printf("\t\tAvailable projections: %d (max).\n",tomos.num_proj);
    }
    else if( verbosity != VERBOSITY_MINIMAL ) {
        printf("    - %d Particles (%d classes) in %s tomograms with max %d projections.\n",
               ptcls.n_ptcl,ptcls.n_refs,tomo_str,tomos.num_proj);
    }

    if( n_orphan > 0 && verbosity != VERBOSITY_MINIMAL )
        printf("    - WARNING: %d particles reference tomograms missing from the tomograms file.\n",n_orphan);
}

inline void print_data_info(Particles*ptcls,Tomograms&tomos,int verbosity) {
    print_data_info(*ptcls,tomos,verbosity);
}

inline void print_loading(int verbosity) {
    if( verbosity != VERBOSITY_MINIMAL ) {
        printf("\tLoading data files...");
        fflush(stdout);
    }
}

inline void print_loaded(int verbosity) {
    if( verbosity != VERBOSITY_MINIMAL ) {
        printf(" Done\n");
        fflush(stdout);
    }
}

inline void exit_bad_args() {
    fprintf(stderr,"Error parsing input arguments.\n");
    exit(1);
}

}

#endif /// DATA_INFO_H
