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

#ifndef REFS_ALIGNER_ARGS_H
#define REFS_ALIGNER_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "gpu.h"
#include "angles_provider.h"
#include "math_cpu.h"
#include "points_provider.h"
#include "reference.h"
#include "arg_parser.h"

namespace ArgsRefsAli {

typedef struct {
    uint32 gpu_ix;
    uint32 box_size;
    single fpix_min;
    single fpix_max;
    single fpix_roll;
    float  cone_range;
    float  cone_step;
    float  inplane_range;
    float  inplane_step;
    uint32 refine_level;
    uint32 refine_factor;
    uint32 off_type;
    float  off_x;
    float  off_y;
    float  off_z;
    float  off_s;
    float  pix_size;

    char   refs_file[SUSAN_FILENAME_LENGTH];
    char   ptcls_out[SUSAN_FILENAME_LENGTH];
    char   ptcls_in [SUSAN_FILENAME_LENGTH];
} Info;

inline bool validate(const Info&info) {
    bool rslt = true;
    if( info.fpix_min >= info.fpix_max ) {
        fprintf(stderr,"Invalid bandpass range: %f - %f.\n",info.fpix_min,info.fpix_max);
        rslt = false;
    }
    if( !IO::exists(info.ptcls_in) ) {
        fprintf(stderr,"Particles file %s does not exist.\n",info.ptcls_in);
        rslt = false;
    }
    if( !IO::exists(info.refs_file) ) {
        fprintf(stderr,"References file %s does not exist.\n",info.refs_file);
        rslt = false;
    }
    else {
        References refs(info.refs_file);
        if( !refs.check_size(info.box_size,true) ) {
            exit(1);
        }
    }
    if( strlen(info.ptcls_in) == 0 ) {
        fprintf(stderr,"Particles output is missing.\n");
        rslt = false;
    }

    int available_gpus = GPU::count_devices();
    if(available_gpus==0) {
        fprintf(stderr,"Not available GPUs on the system.\n");
        rslt = false;
    }
    else {
        if( info.gpu_ix >= available_gpus ) {
            fprintf(stderr,"Requesting unavailable GPU ID %d.\n",info.gpu_ix);
            rslt = false;
        }
    }

    return rslt;
}

inline bool parse_args(Info&info,int ac,char** av) {
    /// Default values:
    info.gpu_ix        = 0;
    info.box_size      = 200;
    info.fpix_min      = 0;
    info.fpix_max      = 30;
    info.fpix_roll     = 4;
    info.cone_range    = 0;
    info.cone_step     = 1;
    info.inplane_range = 0;
    info.inplane_step  = 1;
    info.refine_level  = 0;
    info.refine_factor = 1;
    info.off_type      = ELLIPSOID;
    info.off_x         = 0;
    info.off_y         = 0;
    info.off_z         = 0;
    info.off_s         = 1;
    info.pix_size      = 1;
    memset(info.refs_file,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_out,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_in ,0,SUSAN_FILENAME_LENGTH*sizeof(char));

    /// Parse inputs:
    enum {
        PTCLS_OUT,
        PTCLS_IN,
        REFS_FILE,
        GPU_ID,
        BOX_SIZE,
        BANDPASS,
        ROLLOFF_F,
        CONE,
        INPLANE,
        REFINE,
        OFF_TYPE,
        OFF_PARAM,
        PIX_SIZE
    };

    int c;
    static struct option long_options[] = {
        {"ptcls_in",    1, 0, PTCLS_IN  },
        {"ptcls_out",   1, 0, PTCLS_OUT },
        {"refs_file",   1, 0, REFS_FILE },
        {"gpu_id",      1, 0, GPU_ID    },
        {"box_size",    1, 0, BOX_SIZE  },
        {"bandpass",    1, 0, BANDPASS  },
        {"rolloff_f",   1, 0, ROLLOFF_F },
        {"cone",        1, 0, CONE      },
        {"inplane",     1, 0, INPLANE   },
        {"refine",      1, 0, REFINE    },
        {"off_type",    1, 0, OFF_TYPE  },
        {"off_params",  1, 0, OFF_PARAM },
        {"pix_size",    1, 0, PIX_SIZE  },
        {0, 0, 0, 0}
    };
    
    while( (c=getopt_long_only(ac, av, "", long_options, 0)) >= 0 ) {
        switch(c) {
            case PTCLS_OUT:
                strcpy(info.ptcls_out,optarg);
                break;
            case PTCLS_IN:
                strcpy(info.ptcls_in,optarg);
                break;
            case REFS_FILE:
                strcpy(info.refs_file,optarg);
                break;
            case BOX_SIZE:
                info.box_size = ArgParser::get_even_number(optarg);
                break;
            case GPU_ID:
                info.gpu_ix = atoi(optarg);
                break;
            case BANDPASS:
                ArgParser::get_single_pair(info.fpix_min,info.fpix_max,optarg);
                break;
            case ROLLOFF_F:
                info.fpix_roll = atof(optarg);
                break;
            case CONE:
                ArgParser::get_single_pair(info.cone_range,info.cone_step,optarg);
                break;
            case INPLANE:
                ArgParser::get_single_pair(info.inplane_range,info.inplane_step,optarg);
                break;
            case REFINE:
                ArgParser::get_uint32_pair(info.refine_factor,info.refine_level,optarg);
                break;
            case OFF_TYPE:
                info.off_type = ArgParser::get_offset_type(optarg);
                break;
            case OFF_PARAM:
                ArgParser::get_single_quad(info.off_x,info.off_y,info.off_z,info.off_s,optarg);
                break;
            case PIX_SIZE:
                info.pix_size = atof(optarg);
                break;
            default:
                printf("Unknown parameter %d\n",c);
                exit(1);
                break;
        } /// switch
    } /// while(c)

    return validate(info);
}

inline void print_angles(const Info&info) {

    AnglesProvider angles;
    angles.cone_range    = info.cone_range;
    angles.cone_step     = info.cone_step;
    angles.inplane_range = info.inplane_range;
    angles.inplane_step  = info.inplane_step;
    angles.refine_level  = info.refine_level;
    angles.refine_factor = info.refine_factor;

    int count_total=0;
    int count_level=0;
    int lvl = 0;

    for( angles.levels_init(); angles.levels_available(); angles.levels_next(), lvl++ ) {
        count_level=0;
        for( angles.sym_init(); angles.sym_available(); angles.sym_next() ) {
            for( angles.cone_init(); angles.cone_available(); angles.cone_next() ) {
                for( angles.inplane_init(); angles.inplane_available(); angles.inplane_next() ) {
                    count_level++;
                } /// inplane
            } /// cone
        } /// symmetry
        fprintf(stdout,"\t\tAngles in refinement level %2d: %7d.\n",lvl,count_level);
        count_total += count_level;
    } /// level
    fprintf(stdout,"\t\tTotal angles: %d.\n",count_total);
}

inline void print(const Info&info,FILE*fp=stdout) {
    fprintf(stdout,"\tHalf-set volumes alignment\n");

    fprintf(stdout,"\t\tParticles in:   %s.\n",info.ptcls_in);
    fprintf(stdout,"\t\tReference file: %s.\n",info.refs_file);
    fprintf(stdout,"\t\tParticles out:  %s.\n",info.ptcls_out);

    fprintf(stdout,"\t\tVolume size: %dx%dx%d. (Voxel size: %.3f)\n",info.box_size,info.box_size,info.box_size,info.pix_size);
    
    fprintf(stdout,"\t\tUsing 1 GPU (GPU id: %d).\n",info.gpu_ix);

    fprintf(stdout,"\t\tBandpass: [%.1f - %.1f] fourier pixels",info.fpix_min,info.fpix_max);
    if( info.fpix_roll > 0 )
        fprintf(stdout," with a roll off of %.2f.\n",info.fpix_roll);
    else
        fprintf(stdout,".\n");

    fprintf(stdout,"\t\tCone search:    Range=%.3f, Step=%.3f.\n",info.cone_range,info.cone_step);
    fprintf(stdout,"\t\tInplane search: Range=%.3f, Step=%.3f.\n",info.inplane_range,info.inplane_step);
    fprintf(stdout,"\t\tAngle refinement: Levels=%d, Factor=%d.\n",info.refine_level,info.refine_factor);
    print_angles(info);

    uint32_t total_points=0;
    if( info.off_type == ELLIPSOID ) {
        Vec3*pt = PointsProvider::ellipsoid(total_points,info.off_x,info.off_y,info.off_z,info.off_s);
        fprintf(stdout,"\t\tEllipsoid offset search (3D): ");
        delete [] pt;
    }
    if( info.off_type == CYLINDER ) {
        Vec3*pt = PointsProvider::cylinder(total_points,info.off_x,info.off_y,info.off_z,info.off_s);
        fprintf(stdout,"\t\tCylindrical offset search (3D): ");
        delete [] pt;
    }

    fprintf(stdout,"Range=[%.2f,%.2f,%.2f], Step=%.2f. Total points: %d\n",info.off_x,info.off_y,info.off_z,info.off_s,total_points);
}

}

#endif /// REFS_ALIGNER_ARGS_H

