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

#ifndef REF_MAPS_H
#define REF_MAPS_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#include "datatypes.h"
#include "reference.h"
#include "io.h"
#include "mrc.h"
#include "math_cpu.h"
#include "memory.h"

class RefMap {
public:
    single *map;
    single *half_A;
    single *half_B;
    single *mask;
    single pix_size;
    uint32 box_size;
    uint32 numel;

    bool   has_half_maps() { return (half_A != NULL) && (half_B != NULL); }
    bool   has_ref_map()   { return (map    != NULL); }
    bool   has_ref_mask()  { return (mask   != NULL); }

public:
    RefMap() {
        map    = NULL;
        half_A = NULL;
        half_B = NULL;
        mask   = NULL;

        box_size = 0;
        numel    = 0;
        pix_size = 0;
    }

    ~RefMap() {
        free_array(map   );
        free_array(half_A);
        free_array(half_B);
        free_array(mask  );
    }

    void load(const Reference&ref_info,bool apply_mask=true) {

        box_size = get_box_size(ref_info.map);

        if( box_size > 0 ) {

            pix_size = Mrc::get_apix(ref_info.map);
            numel = box_size*box_size*box_size;
            map   = load_mrc(ref_info.map);

            if( get_box_size(ref_info.h1) == box_size )
                half_A = load_mrc(ref_info.h1);

            if( get_box_size(ref_info.h2) == box_size )
                half_B = load_mrc(ref_info.h2);

            if( get_box_size(ref_info.mask) == box_size ) {
                mask = load_mrc(ref_info.mask);
            }

            if( mask != NULL && apply_mask ) {
                if( map != NULL ) {
                    if( !Math::normalize_masked(map,mask,numel) ) {
                        fprintf(stderr,"Error normalizing map %s\n",ref_info.map);
                        exit(1);
                    }
                }

                if( half_A != NULL ) {
                    if( !Math::normalize_masked(half_A,mask,numel) ) {
                        fprintf(stderr,"Error normalizing map %s\n",ref_info.h1);
                        exit(1);
                    }
                }

                if( half_B != NULL ) {
                    if( !Math::normalize_masked(half_B,mask,numel) ) {
                        fprintf(stderr,"Error normalizing map %s\n",ref_info.h2);
                        exit(1);
                    }
                }
            }

        }
    }

    /// Local normalization radius (A): outermost integer shell from the box centre whose rms of the
    /// masked map is at least SUSAN_LCN_PROFILE_FRAC of the maximum shell rms, times the pixel size.
    float local_radius() {
        const single*v = (map != NULL)? map : half_A;
        if( v == NULL || box_size == 0 )
            return 0;
        const int N = (int)box_size;
        const int ns = (int)lroundf(sqrtf(3.0f)*(N/2)) + 2;
        std::vector<double> sum(ns,0.0);
        std::vector<long>   cnt(ns,0);
        for(int z=0;z<N;z++) {
            for(int y=0;y<N;y++) {
                for(int x=0;x<N;x++) {
                    const float dx = x-N/2, dy = y-N/2, dz = z-N/2;
                    const int sh = (int)lroundf(sqrtf(dx*dx+dy*dy+dz*dz));
                    const double val = v[((size_t)z*N+y)*N+x];
                    sum[sh] += val*val;
                    cnt[sh]++;
                }
            }
        }
        std::vector<double> prof(ns,0.0);
        double mx = 0;
        for(int i=0;i<ns;i++) {
            if( cnt[i] > 0 ) prof[i] = sqrt(sum[i]/cnt[i]);
            mx = std::max(mx,prof[i]);
        }
        int outer = 0;
        for(int i=0;i<ns;i++)
            if( prof[i] >= SUSAN_LCN_PROFILE_FRAC*mx ) outer = i;
        return outer*pix_size;
    }

private:
    uint32 get_box_size(const char*fn) {

        uint32 X=0,Y=0,Z=0;

        if( check_mrc_exists(fn) ) {
            Mrc::read_size(X,Y,Z,fn);
            if( (X == Y) && (Y == Z) )
                return X;
            else
                return 0;
        }
        else
            return 0;
    }

    bool   check_mrc_exists(const char*fn) {
        bool rslt = false;
        if( fn[0] != 0 )
            if( IO::exists(fn) )
                rslt = true;
        return rslt;
    }

    single *load_mrc(const char*fn) {
        uint32 X=0,Y=0,Z=0;
        return Mrc::read(X,Y,Z,fn);
    }
};

/// Prints the local normalization radii (A), once.
inline void print_local_radii(const std::vector<float>&radii) {
    printf("    Local normalization reference radius (A):");
    for(size_t r=0;r<radii.size();r++)
        printf(" %.1f",radii[r]);
    printf("\n");
}

#endif /// REF_MAPS_H


