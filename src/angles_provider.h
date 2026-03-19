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

#ifndef ANGLES_PROVIDER_H
#define ANGLES_PROVIDER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "datatypes.h"
#include "math_cpu.h"
#include "angles_symmetry.h"
#include "memory.h"

using namespace Eigen;
using namespace Math;

class AnglesProvider {

public:
    float  cone_range;
    float  cone_step;
    float  inplane_range;
    float  inplane_step;
    uint32 refine_level;
    uint32 refine_factor;

protected:
    M33f   *pseudo_sym_list;
    uint32 pseudo_sym_length;

    uint32 curr_lvl;
    uint32 curr_sym;

    double c_stp;
    double c_end;
    uint32 c_ite;
    uint32 c_lim;

    double ip_ini;
    double ip_stp;
    double ip_end;

    double eu1;
    double eu2;
    double eu3;

public:
    AnglesProvider() {
        cone_range    = 0;
        cone_step     = 1;
        inplane_range = 0;
        inplane_step  = 1;
        refine_level  = 0;
        refine_factor = 2;

        curr_lvl = 0;
        curr_sym = 0;

        pseudo_sym_list = AnglesSymmetry::get_rotation_list(pseudo_sym_length,"c1");
    }

    ~AnglesProvider() {
        free_array(pseudo_sym_list);
    }

    int max_num_angles_any_level() {
        int count_max  =0;
        int count_level=0;
        for( levels_init(); levels_available(); levels_next() ) {
            count_level=0;
            for( sym_init(); sym_available(); sym_next() ) {
                for( cone_init(); cone_available(); cone_next() ) {
                    for( inplane_init(); inplane_available(); inplane_next() ) {
                        count_level++;
                    } /// inplane
                } /// cone
            } /// symmetry
            if( count_level > count_max )
                count_max = count_level;
        }
        return count_max;
    }

    void set_symmetry(const char*sym_type) {
        free_array(pseudo_sym_list);
        pseudo_sym_list = AnglesSymmetry::get_rotation_list(pseudo_sym_length,sym_type);
        if( pseudo_sym_list == NULL ) {
            fprintf(stderr,"[ERROR] AnglesSymmetry: Unknown/invalid symmetry: %s.\n",sym_type);
            exit(1);
        }
    }

    void levels_init() {
        curr_lvl = 0;

        c_stp = cone_step;
        c_end = cone_range/2;

        ip_stp = get_angle_step(inplane_range, inplane_step);

        float i_s = ip_stp*floor(inplane_range/(2*ip_stp));

        ip_ini = -i_s;
        ip_end =  i_s;

        if( !(ip_end<180.0) )
            ip_end = ip_end - ip_stp;
    }

    bool levels_available() {
        return ( curr_lvl <= refine_level );
    }

    void levels_next() {
        curr_lvl++;
        if( cone_range > 0 ) {
            c_end = refine_factor*c_stp/2;
            c_stp = cone_step;
            for(uint32 i=0;i<curr_lvl;i++) c_stp = c_stp/2;
        }
        if( inplane_range > 0 ) {
            double rf = (double)refine_factor;
            ip_ini = -rf*ip_stp/2;
            ip_end =  rf*ip_stp/2;
            ip_stp = inplane_step;
            for(uint32 i=0;i<curr_lvl;i++) ip_stp = ip_stp/2;
        }
    }

    void sym_init() {
        curr_sym = 0;
    }

    bool sym_available() {
        if( curr_lvl > 0 )
            return ( curr_sym < 1 );
        else
            return ( curr_sym < pseudo_sym_length );
    }

    void sym_next() {
        curr_sym++;
    }

    void cone_init() {
        eu1 = 0;
        eu2 = 0;
        c_ite = 0;
        c_lim = 0;
    }

    bool cone_available() {
        return (eu2 <= c_end);
    }

    void cone_next() {
        c_ite++;
        if( c_ite >= c_lim ) {
            c_ite = 0;
            eu2 += c_stp;
            c_lim = (uint32)fmax(round( 360*sin(eu2*M_PI/180.0f)/c_stp ),1.0);
        }
        if( c_lim > 0 )
            eu1 = 360*(double)c_ite/(double)c_lim;
        else
            eu1 = 0;
    }

    void inplane_init() {
        eu3 = ip_ini;
    }

    bool inplane_available() {
        return (eu3 <= ip_end);
    }

    void inplane_next() {
        eu3 += ip_stp;
    }

    void print_lvl_angle() {
        printf("lvl: %d: %7.2f:%7.2f:%7.2f | %7.2f:%7.2f:%7.2f\n",curr_lvl,0,c_stp,c_end,ip_ini,ip_stp,ip_end);
    }

    void print_curr_angles() {
        M33f R;
        V3f eu;
        get_current_R(R);
        Rmat_eZYZ(eu,R);
        eu *= RAD2DEG;
        printf("    %d: [ %7.2f %7.2f %7.2f ]\n",curr_lvl,eu(0),eu(1),eu(2));
    }

    void get_current_R(M33f&R) {
        /// Sample Cone trying to keep the Y axis always point forward
        /// (avoid twisting from the cone sampling)
        float phi   = eu1 * DEG2RAD;
        float theta = eu2 * DEG2RAD;
        float psi   = eu3 * DEG2RAD;

        // current forward
        V3f forward;
        forward << sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta);

        // Direction trying to keep
        V3f face_direction(0,1,0);

        if (fabs(forward.dot(face_direction)) > 0.999f)
            face_direction = V3f(1,0,0);

        // Be sure to continue looking forward
        V3f right = face_direction.cross(forward).normalized();
        V3f up    = forward.cross(right);

        M33f Rcone;
        Rcone.col(0) = right;
        Rcone.col(1) = up;
        Rcone.col(2) = forward;

        // Add in plane rotations
        Eigen::AngleAxisf rz(psi, forward);
        M33f Rinplane = rz.toRotationMatrix();

        // Final rotation
        R = Rcone*Rinplane*pseudo_sym_list[curr_sym];
    }

    void get_current_R(Rot33&R) {
        M33f Rtmp;
        get_current_R(Rtmp);
        Math::set(R,Rtmp);
    }

protected:
    double get_angle_step(const double range, const double step) {
        double rslt = 1.0f;
        if( range > SUSAN_FLOAT_TOL ) {
            rslt = range / round(range/step);
        }
        return rslt;
    }

};

#endif /// ANGLES_PROVIDER_H



