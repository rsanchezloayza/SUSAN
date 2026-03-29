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

#ifndef CROP_PROJECTIONS_ARGS_H
#define CROP_PROJECTIONS_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "io.h"
#include "arg_parser.h"

namespace ArgsCropProjections {

typedef struct {
    uint32 n_threads;
    uint32 box_size;
    uint32 norm_type;
    uint32 out_fmt;
    bool   invert;

    char   out_dir[SUSAN_FILENAME_LENGTH];
    char   ptcls_in[SUSAN_FILENAME_LENGTH];
    char   tomos_in[SUSAN_FILENAME_LENGTH];
} Info;

inline bool validate(const Info&info) {
    bool rslt = true;
    if( !IO::exists(info.ptcls_in) ) {
        fprintf(stderr,"Particles file %s does not exist.\n",info.ptcls_in);
        rslt = false;
    }
    if( !IO::exists(info.tomos_in) ) {
        fprintf(stderr,"Tomos file %s does not exist.\n",info.tomos_in);
        rslt = false;
    }
    if( strlen(info.out_dir) == 0 ) {
        fprintf(stderr,"Output folder missing.\n");
        rslt = false;
    }
    return rslt;
};

inline bool parse_args(Info&info,int ac,char** av) {
    /// Default values:
    info.n_threads   = 1;
    info.box_size    = 200;
    info.norm_type   = NO_NORM;
    info.out_fmt     = CROP_MRC;
    info.invert      = false;
    memset(info.out_dir ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_in,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.tomos_in,0,SUSAN_FILENAME_LENGTH*sizeof(char));

    /// Parse inputs:
    enum {
        TOMOS_FILE,
        OUT_FOLDER,
        PTCLS_FILE,
        N_THREADS,
        BOX_SIZE,
        NORM_TYPE,
        FORMAT,
        INVERT
    };

    int c;
    static struct option long_options[] = {
        {"tomos_file",  1, 0, TOMOS_FILE },
        {"out_dir",     1, 0, OUT_FOLDER },
        {"ptcls_file",  1, 0, PTCLS_FILE },
        {"n_threads",   1, 0, N_THREADS  },
        {"box_size",    1, 0, BOX_SIZE   },
        {"norm_type",   1, 0, NORM_TYPE  },
        {"format",      1, 0, FORMAT     },
        {"invert",      1, 0, INVERT     },
        {0, 0, 0, 0}
    };

    while( (c=getopt_long_only(ac, av, "", long_options, 0)) >= 0 ) {
        switch(c) {
            case TOMOS_FILE:
                strcpy(info.tomos_in,optarg);
                break;
            case OUT_FOLDER:
                strcpy(info.out_dir,optarg);
                break;
            case PTCLS_FILE:
                strcpy(info.ptcls_in,optarg);
                break;
            case BOX_SIZE:
                info.box_size = ArgParser::get_even_number(optarg);
                break;
            case N_THREADS:
                info.n_threads = atoi(optarg);
                break;
            case NORM_TYPE:
                info.norm_type = ArgParser::get_norm_type(optarg);
                break;
            case FORMAT:
                info.out_fmt = ArgParser::get_format_output(optarg);
                break;
            case INVERT:
                info.invert = ArgParser::get_bool(optarg);
                break;
            default:
                printf("Unknown parameter %d\n",c);
                exit(1);
                break;
        } /// switch
    } /// while(c)

    return validate(info);
}

inline void print(const Info&info,FILE*fp=stdout) {
    fprintf(stdout,"\tCropping projections:\n");

    fprintf(stdout,"\t\tParticles file: %s.\n",info.ptcls_in);
    fprintf(stdout,"\t\tTomograms file: %s.\n",info.tomos_in);
    fprintf(stdout,"\t\tOutput folder: %s.\n",info.out_dir);

    fprintf(stdout,"\t\tPatch size: %dx%d\n",info.box_size,info.box_size);

    if( info.n_threads > 1 ) {
        fprintf(stdout,"\t\tUsing %d threads.\n",info.n_threads);
    }
    else{
        fprintf(stdout,"\t\tUsing 1 thread.\n");
    }

    if( info.norm_type == NO_NORM )
        fprintf(stdout,"\t\tSubstack normalization policy: Disabled.\n");
    if( info.norm_type == ZERO_MEAN )
        fprintf(stdout,"\t\tSubstack normalization policy: Mean=0.\n");
    if( info.norm_type == ZERO_MEAN_1_STD )
        fprintf(stdout,"\t\tSubstack normalization policy: Mean=0, Std=1.\n");
    if( info.norm_type == ZERO_MEAN_W_STD )
        fprintf(stdout,"\t\tSubstack normalization policy: Mean=0, Std according to projection weight.\n");
}


}

#endif /// CROP_PROJECTIONS_ARGS_H

