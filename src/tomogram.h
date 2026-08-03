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

#ifndef TOMOGRAM_H
#define TOMOGRAM_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "datatypes.h"
#include "memory.h"
#include "io.h"
#include "math_cpu.h"
#include "mrc.h"

class Tomogram {
public:
    uint32  tomo_id;
    VUInt3  tomo_dim;
    V3f     tomo_center;
    V3f     tomo_position;
    char    stk_name[ SUSAN_FILENAME_LENGTH ];
    VUInt3  stk_dim;
    V3f     stk_center;
    uint32  num_proj;
    single  pix_size;
    single  KV;
    single  CS;
    single  AC;
    single  handedness;
    M33f    *R;
    V3f     *t;
    single  *w;
    Defocus *def;
    single  *doses;
    single  *nml_tilt_ang;
    single  *ctf_scale;

public:
    Tomogram() {
        tomo_id       = 0;
        tomo_dim.x    = 0;
        tomo_dim.y    = 0;
        tomo_dim.z    = 0;
        stk_name[0]   = 0;
        stk_dim.x     = 0;
        stk_dim.y     = 0;
        stk_dim.z     = 0;
        num_proj      = 0;
        pix_size      = 1;
        KV            = 300;
        CS            = 2.7;
        AC            = 0.07;
        handedness    = -1.0;
        R             = NULL;
        t             = NULL;
        w             = NULL;
        def           = NULL;
        doses         = NULL;
        nml_tilt_ang  = NULL;
        ctf_scale     = NULL;
    }

    ~Tomogram() {
        free_array( R   );
        free_array( t   );
        free_array( w   );
        free_array( def );
        free_array(doses);
        free_array(nml_tilt_ang);
        free_array(ctf_scale);
    }

    void read(IO::TxtParser&parser) {

        bool extended_params = false;
        Vec3 pos_buffer;

        parser.parse_args();
        parser.args_get_val( tomo_id    ,"tomo_id"   );
        parser.args_get_val( tomo_dim   ,"tomo_size" );
        parser.args_get_val( stk_name   ,"stack_file");
        parser.args_get_val( stk_dim    ,"stack_size");
        parser.args_get_val( pix_size   ,"pix_size"  );
        parser.args_get_val( KV         ,"kv"        );
        parser.args_get_val( CS         ,"cs"        );
        parser.args_get_val( AC         ,"ac"        );
        parser.args_get_val( num_proj   ,"num_proj"  );
        parser.args_get_val( pos_buffer ,"tomo_pos"  );
        extended_params = parser.args_get_val( handedness,"handedness",-1);

        tomo_position(0) = pos_buffer.x;
        tomo_position(1) = pos_buffer.y;
        tomo_position(2) = pos_buffer.z;

        tomo_center(0) = ((float)tomo_dim.x)/2;
        tomo_center(1) = ((float)tomo_dim.y)/2;
        tomo_center(2) = ((float)tomo_dim.z)/2;

        stk_center(0) = ((float)stk_dim.x)/2;
        stk_center(1) = ((float)stk_dim.y)/2;
        stk_center(2) = ((float)stk_dim.z)/2;

        allocate(num_proj);

        if( extended_params )
            parse_extended_params(parser);
        else
            parse_params(parser);

    }

    void print() {
        printf("  Tomo ID:    %d\n",tomo_id);
        printf("  Tomo size:  [%d %d %d]\n",tomo_dim.x,tomo_dim.y,tomo_dim.z);
        printf("  Stack file: %s\n",stk_name);
        printf("  Stack size: [%d %d %d]\n",stk_dim.x,stk_dim.y,stk_dim.z);
        printf("  Num Proj:   %d\n",num_proj);
        printf("  Pix size:   %.3f\n",pix_size);
        printf("  KV:         %.1f\n",KV);
        printf("  CS:         %.2f\n",CS);
        printf("  AC:         %.3f\n",AC);
        printf("  Handedness: %.3f\n",handedness);
        printf("  Projections:\n");
        for(int i=0;i<num_proj;i++) {
            printf("    %2d:  %7.4f %7.4f %7.4f   %7.2f\n",      i,R[i](0,0),R[i](0,1),R[i](0,2),t[i](0));
            printf( "         %7.4f %7.4f %7.4f   %7.2f\n",        R[i](1,0),R[i](1,1),R[i](1,2),t[i](1));
            printf( "         %7.4f %7.4f %7.4f             %f\n", R[i](2,0),R[i](2,1),R[i](2,2),w[i]);
        }
        printf("  Defocus:\n");
        for(int i=0;i<num_proj;i++) {
            printf("    %2d:  %10.2f %10.2f %8.3f %7.2f %7.2f %8.4f\n",i,def[i].U,def[i].V,def[i].angle,def[i].Bfactor,def[i].ExpFilt,def[i].max_res);
        }
        printf("  Doses:\n");
        for(int i=0;i<num_proj;i++) {
            printf("    %2d:  %8.4f\n",i,doses[i]);
        }
        printf("  Nominal Tilts Angles:\n");
        for(int i=0;i<num_proj;i++) {
            printf("    %2d:  %8.4f\n",i,nml_tilt_ang[i]);
        }
        printf("  CTF Scale Factor:\n");
        for(int i=0;i<num_proj;i++) {
            printf("    %2d:  %8.4f\n",i,ctf_scale[i]);
        }
    }
    
    /// If stk_name is a relative path that cannot be found from the current
    /// working directory, try to resolve it relative to base_dir (the location
    /// of the tomostxt file). Absolute paths are left untouched.
    void resolve_stack_path(const char*base_dir) {
        if( stk_name[0] == '/'      ) return;
        if( IO::exists( stk_name  ) ) return;

        char buffer[ SUSAN_FILENAME_LENGTH ];
        int n = snprintf(buffer,SUSAN_FILENAME_LENGTH,"%s/%s",base_dir,stk_name);

        if( n > 0 && n < SUSAN_FILENAME_LENGTH ) {
            if( IO::exists(buffer) )
                strcpy(stk_name,buffer);
        }
    }

    bool check() {
        bool rslt = check_exists();

        if( rslt ) {
            rslt &= check_size();
            rslt &= check_mode();
            rslt &= check_apix();
        }

        return rslt;
    }

    float get_angle_step_rad() {
        float angle_step = 0;
        int angle_count = 0;

        float prv_ang = 0;
        for(int j=0;j<num_proj;j++) {
            float cur_ang = Math::get_Y_angle_rad(R[j]);
            if( j>0 ) {
                angle_step += fabs(cur_ang-prv_ang);
                angle_count++;
            }
            prv_ang = cur_ang;
        }

        return angle_step/angle_count;
    }

protected:
    void allocate(int K) {
        R   = new M33f   [K];
        t   = new V3f    [K];
        w   = new float  [K];
        def = new Defocus[K];
        doses = new float[K];
        nml_tilt_ang = new float[K];
        ctf_scale = new float[K];
    }
    
    bool check_exists() {
        if( !IO::exists(stk_name) ) {
            fprintf(stderr,"File %s not found or cannot be read.\n",stk_name);
            return false;
        }
        return true;
    }

    bool check_size() {
        uint32 x,y,z;
        Mrc::read_size(x,y,z,stk_name);
        if( x != stk_dim.x || y != stk_dim.y || z != stk_dim.z ) {
            fprintf(stderr,"file %s: Different size (%d,%d,%d) != (%d,%d,%d).\n",stk_name,x,y,z,stk_dim.x,stk_dim.y,stk_dim.z);
            return false;
        }
        return true;
    }

    bool check_mode() {
        if( !Mrc::is_mode_float(stk_name) ) {
            fprintf(stderr,"file %s: Unsupported MRC mode.\n",stk_name);
            return false;
        }
        return true;
    }

    bool check_apix() {
        float stk_pix_size = Mrc::get_apix(stk_name);
        if( abs(stk_pix_size-pix_size) < SUSAN_FLOAT_TOL ) {
            fprintf(stderr,"file %s: Different pixel size %f != %f.\n",stk_name,stk_pix_size,pix_size);
            return false;
        }
        return true;
    }

    void parse_params(IO::TxtParser&parser) {

        int i=0;
        while( parser.get_next_line() ) {
            if(i<num_proj) {
                V3f eu;
                t[i](2) = 0;
                int n = sscanf(parser.get_buffer(),"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
                               &eu(0),&eu(1),&eu(2),&t[i](0),&t[i](1),&w[i],
                               &def[i].U,&def[i].V,&def[i].angle,&def[i].ph_shft,
                               &def[i].Bfactor,&def[i].ExpFilt,
                               &def[i].max_res,&def[i].score);

                if( n != 14 ) {
                    fprintf(stderr,"Truncated tomogram file.\n");
                    exit(1);
                }

                eu *= DEG2RAD;
                Math::eZYZ_Rmat(R[i],eu);

            }
            i++;
        }

        if( i != num_proj ) {
            fprintf(stderr,"Error parsing tomogram data.\n");
            exit(1);
        }
    }

    void parse_extended_params(IO::TxtParser&parser) {

        int i=0;
        while( parser.get_next_line() ) {
            if(i<num_proj) {
                V3f eu;
                t[i](2) = 0;

                int n = sscanf(parser.get_buffer(),"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",                           &eu(0),&eu(1),&eu(2),&t[i](0),&t[i](1),&w[i],
                               &def[i].U,&def[i].V,&def[i].angle,&def[i].ph_shft,
                               &def[i].Bfactor,&def[i].ExpFilt,
                               &def[i].max_res,&def[i].score,
                               &doses[i],&nml_tilt_ang[i],&ctf_scale[i]);

                if( n != 17 ) {
                    fprintf(stderr,"Truncated tomogram file.\n");
                    exit(1);
                }

                eu *= DEG2RAD;
                Math::eZYZ_Rmat(R[i],eu);

            }
            i++;
        }

        if( i != num_proj ) {
            fprintf(stderr,"Error parsing tomogram data.\n");
            exit(1);
        }
    }
};

class Tomograms {
public:
    uint32   num_tomo;
    uint32   num_proj;

protected:
    Tomogram *tomos;

public:
    Tomograms(const char*filename) {
        tomos = NULL;

        char base_dir[ SUSAN_FILENAME_LENGTH ];
        IO::get_dirname(base_dir,filename);

        IO::TxtParser parser(filename,"tomostxt");
        parser.parse_args();
        parser.args_get_val(num_tomo,"num_tomos");
        parser.args_get_val(num_proj,"num_projs");

        tomos = new Tomogram[num_tomo];

        for(int i=0;i<num_tomo;i++) {
            tomos[i].read(parser);
            tomos[i].resolve_stack_path(base_dir);
        }

    }

    ~Tomograms() {
        free_array(tomos);
    }

    Tomogram& operator[](int ix) {
        if( ix >= num_tomo ) {
            fprintf(stderr,"Trying to access tomogram %d from %d.\n",ix,num_tomo);
            exit(0);
        }
        return tomos[ix];
    }

    Tomogram& at(int ix) {
        if( ix >= num_tomo ) {
            fprintf(stderr,"Trying to access tomogram %d from %d.\n",ix,num_tomo);
            exit(0);
        }
        return tomos[ix];
    }

    void print() {
        printf("Num Tomos: %d\n",num_tomo);
        printf("Max Projs: %d\n",num_proj);
        for(int i=0;i<num_tomo;i++) {
            printf("Tomogram %d\n",i);
            tomos[i].print();
        }
    }
    
    bool check() {
        bool rslt = true;
        for(int i=0;i<num_tomo;i++) {
            rslt &= tomos[i].check();
        }
        return rslt;
    }

};

#endif /// TOMOGRAM_H


