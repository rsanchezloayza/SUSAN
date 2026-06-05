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

#endif /// REF_MAPS_H


