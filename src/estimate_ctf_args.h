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

#ifndef ESTIMATE_CTF_PARSE_H
#define ESTIMATE_CTF_PARSE_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "gpu.h"
#include "arg_parser.h"

namespace ArgsCTF {

typedef struct {
    uint32 n_threads;
    uint32 box_size;
    single res_max;
    single def_min;
    single def_max;
    single res_thres;
    bool   is_overfocus;
    bool   est_phase_shift;
    bool   est_initial_snr;
    int    verbosity;
    int    log_level;
    int    n_gpu;
    uint32 p_gpu[SUSAN_MAX_N_GPU];
    char   out_dir[SUSAN_FILENAME_LENGTH];
    char   ptcls_in[SUSAN_FILENAME_LENGTH];
    char   tomos_in[SUSAN_FILENAME_LENGTH];
} Info;

inline bool validate(const Info&info) {
    bool rslt = true;
    if( info.res_max <= 0 ) {
        fprintf(stderr,"Invalid maximum resolution: %f.\n",info.res_max);
        rslt = false;
    }
    if( info.def_max <= info.def_min ) {
        fprintf(stderr,"Invalid defocus range: %f - %f.\n",info.def_min,info.def_max);
        rslt = false;
    }
    if( !IO::exists(info.ptcls_in) ) {
        fprintf(stderr,"Particles file %s does not exist.\n",info.ptcls_in);
        rslt = false;
    }
    if( !IO::exists(info.tomos_in) ) {
        fprintf(stderr,"Tomos file %s does not exist.\n",info.tomos_in);
        rslt = false;
    }
    IO::create_dir(info.out_dir);
    if( !IO::exist_dir(info.out_dir) ) {
        fprintf(stderr,"Output folder %s cannot be created.\n",info.out_dir);
        rslt = false;
    }
    if( !GPU::check_gpu_id_list(info.n_gpu,info.p_gpu) ) {
        fprintf(stderr,"Error with CUDA devices.\n");
        rslt = false;
    }

    return rslt;
}

inline bool parse_args(Info&info,int ac,char** av) {
    /// Default values:
    info.n_threads = 1;
    info.box_size  = 512;
    info.res_thres = 0.5;
    info.res_max   = 7;
    info.def_min   = 0;
    info.def_max   = 0;
    info.n_gpu     = 0;
    info.verbosity = VERBOSITY_BASIC;
    info.log_level = 0;
    memset(info.p_gpu   ,0,SUSAN_MAX_N_GPU*sizeof(uint32));
    memset(info.out_dir ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_in,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.tomos_in,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    info.is_overfocus    = true;
    info.est_phase_shift = true;
    info.est_initial_snr = false;

    /// Parse inputs:
    enum {
        TOMOS_IN,
        DATA_OUT,
        RES_MAX,
        RES_THRES,
        DEF_RANGE,
        PTCLS_FILE,
        BOX_SIZE,
        N_THREADS,
        GPU_LIST,
        OVERFOCUS,
        EST_PHASE_SHIFT,
        EST_INITIAL_SNR,
        VERBOSITY,
        LOG_LEVEL
    };

    int c;
    static struct option long_options[] = {
        {"tomos_in",   1, 0, TOMOS_IN  },
        {"data_out",   1, 0, DATA_OUT  },
        {"ptcls_file", 1, 0, PTCLS_FILE},
        {"box_size",   1, 0, BOX_SIZE  },
        {"n_threads",  1, 0, N_THREADS },
        {"res_max",    1, 0, RES_MAX   },
        {"res_thres",  1, 0, RES_THRES },
        {"def_range",  1, 0, DEF_RANGE },
        {"gpu_list",   1, 0, GPU_LIST  },
        {"verbosity",  1, 0, VERBOSITY },
        {"log_level",  1, 0, LOG_LEVEL },
        {"overfocus",      1, 0, OVERFOCUS       },
        {"est_phase_shift",1, 0, EST_PHASE_SHIFT },
        {"est_initial_snr",1, 0, EST_INITIAL_SNR },
        {0, 0, 0, 0}
    };
    
    while( (c=getopt_long_only(ac, av, "", long_options, 0)) >= 0 ) {
        switch(c) {
            case TOMOS_IN:
                strcpy(info.tomos_in,optarg);
                break;
            case DATA_OUT:
                strcpy(info.out_dir,optarg);
                break;
            case PTCLS_FILE:
                strcpy(info.ptcls_in,optarg);
                break;
            case BOX_SIZE:
                info.box_size  = atoi(optarg);
                info.box_size += (info.box_size & 0x01); // Force box to be multiple of 4.
                info.box_size += (info.box_size & 0x02);
                break;
            case N_THREADS:
                info.n_threads = atoi(optarg);
                break;
            case OVERFOCUS:
                info.is_overfocus = atoi(optarg)>0;
                break;
            case EST_PHASE_SHIFT:
                info.est_phase_shift = atoi(optarg)>0;
                break;
            case EST_INITIAL_SNR:
                info.est_initial_snr = atoi(optarg)>0;
                break;
            case GPU_LIST:
                info.n_gpu = ArgParser::get_list_integers(info.p_gpu,optarg);
                break;
            case RES_MAX:
                info.res_max = atof(optarg);
                break;
            case RES_THRES:
                info.res_thres = atof(optarg);
                break;
            case DEF_RANGE:
                ArgParser::get_single_pair(info.def_min,info.def_max,optarg);
                break;
            case VERBOSITY:
                info.verbosity = atoi(optarg);
                break;
            case LOG_LEVEL:
                info.log_level = atoi(optarg);
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
    fprintf(fp,"\tCtf Estimation:\n");

    fprintf(fp,"\t\tParticles file: %s.\n",info.ptcls_in);
    fprintf(fp,"\t\tTomograms file: %s.\n",info.tomos_in);
    fprintf(fp,"\t\tOutput folder: %s.\n",info.out_dir);

    fprintf(fp,"\t\tPatch size: %dx%d.\n",info.box_size,info.box_size);

    if( info.n_gpu > 1 ) {
        fprintf(fp,"\t\tUsing %d GPUs (GPU ids: %d",info.n_gpu,info.p_gpu[0]);
        for(int i=1;i<info.n_gpu;i++)
            fprintf(fp,",%d",info.p_gpu[i]);
        fprintf(fp,"), ");
    }
    else {
        fprintf(fp,"\t\tUsing 1 GPU (GPU id: %d), ",info.p_gpu[0]);
    }

    if( info.n_threads > 1 ) {
        fprintf(fp,"and %d threads.\n",info.n_threads);
    }
    else{
        fprintf(fp,"and 1 thread.\n");
    }

    fprintf(fp,"\t\tMaximum resolution: %.1f Å.\n",info.res_max);
    fprintf(fp,"\t\tResolution threshold: %.2f.\n",info.res_thres);
    fprintf(fp,"\t\tDefocus range: %.2f - %.2f Å.\n",info.def_min,info.def_max);
    fprintf(fp,"\t\tPhase shift estimation: %s.\n",info.est_phase_shift?"enabled":"disabled");
    fprintf(fp,"\t\tInitial SNR estimation: %s.\n",info.est_initial_snr?"enabled":"disabled");
    fprintf(fp,"\t\tDebug data log level: %d.\n",info.log_level);

}

}

#endif /// ESTIMATE_CTF_PARSE_H

