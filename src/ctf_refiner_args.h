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

#ifndef CTF_REFINER_ARGS_H
#define CTF_REFINER_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "gpu.h"
#include "arg_parser.h"
#include "reference.h"

namespace ArgsCtfRef {

typedef struct {
    int    n_gpu;
    uint32 p_gpu[SUSAN_MAX_N_GPU];
    uint32 n_threads;
    uint32 box_size;
    single fpix_min;
    single fpix_max;
    single fpix_roll;
    uint32 pad_size;
    uint32 pad_type;
    uint32 norm_type;
    float  def_range;
    float  def_step;
    float  ang_range;
    float  ang_step;
    float  ssnr_F;
    float  ssnr_S;
    bool   est_dose;
    bool   use_halves;
    bool   astigmatism;
    float  off_x;
    float  off_y;
    float  off_z;
    float  off_s;

    char   refs_file[SUSAN_FILENAME_LENGTH];
    char   ptcls_out[SUSAN_FILENAME_LENGTH];
    char   ptcls_in [SUSAN_FILENAME_LENGTH];
    char   tomo_file[SUSAN_FILENAME_LENGTH];

    int verbosity;
} Info;

bool validate(const Info&info) {
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
        if( !refs.check_fields(info.use_halves) ) {
            exit(1);
        }
        if( !refs.check_size(info.box_size,info.use_halves) ) {
            exit(1);
        }
    }
    if( !IO::exists(info.tomo_file) ) {
        fprintf(stderr,"Tomos file %s does not exist.\n",info.tomo_file);
        rslt = false;
    }
    if( strlen(info.ptcls_out) == 0 ) {
        fprintf(stderr,"Particles output is missing.\n");
        rslt = false;
    }
    if( !GPU::check_gpu_id_list(info.n_gpu,info.p_gpu) ) {
        fprintf(stderr,"Error with CUDA devices.\n");
        rslt = false;
    }

    return rslt;
}

bool parse_args(Info&info,int ac,char** av) {
    /// Default values:
    info.n_gpu       = 0;
    info.n_threads   = 1;
    info.box_size    = 200;
    info.fpix_min    = 5;
    info.fpix_max    = 40;
    info.fpix_roll   = 4;
    info.pad_size    = 0;
    info.pad_type    = PAD_ZERO;
    info.norm_type   = NO_NORM;
    info.def_range   = 1000;
    info.def_step    = 100;
    info.ang_range   = 20;
    info.ang_step    = 5;
    info.verbosity   = 0;
    info.ssnr_F      = 0;
    info.ssnr_S      = 0;
    info.est_dose    = false;
    info.use_halves  = false;
    info.astigmatism = false;
    info.off_x       = 0;
    info.off_y       = 0;
    info.off_z       = 0;
    info.off_s       = 1;

    memset(info.p_gpu    ,0,SUSAN_MAX_N_GPU*sizeof(uint32));
    memset(info.refs_file,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_out,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_in ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.tomo_file,0,SUSAN_FILENAME_LENGTH*sizeof(char));

    /// Parse inputs:
    enum {
        TOMOS_FILE,
        PTCLS_OUT,
        PTCLS_IN,
        REFS_FILE,
        N_THREADS,
        GPU_LIST,
        BOX_SIZE,
        PAD_SIZE,
        PAD_TYPE,
        NORM_TYPE,
        SSNR,
        ASTIGMATISM,
        BANDPASS,
        ROLLOFF_F,
        USE_HALVES,
        DEF_SEARCH,
        ANG_SEARCH,
        VERBOSITY,
        OFF_PARAM,
        EST_DOSE
    };

    int c;
    static struct option long_options[] = {
        {"tomos_file", 1, 0, TOMOS_FILE },
        {"ptcls_in",   1, 0, PTCLS_IN   },
        {"ptcls_out",  1, 0, PTCLS_OUT  },
        {"refs_file",  1, 0, REFS_FILE  },
        {"n_threads",  1, 0, N_THREADS  },
        {"gpu_list",   1, 0, GPU_LIST   },
        {"box_size",   1, 0, BOX_SIZE   },
        {"pad_size",   1, 0, PAD_SIZE   },
        {"pad_type",   1, 0, PAD_TYPE   },
        {"norm_type",  1, 0, NORM_TYPE  },
        {"ssnr_param", 1, 0, SSNR       },
        {"bandpass",   1, 0, BANDPASS   },
        {"rolloff_f",  1, 0, ROLLOFF_F  },
        {"def_search", 1, 0, DEF_SEARCH },
        {"ang_search", 1, 0, ANG_SEARCH },
        {"est_dose",   1, 0, EST_DOSE   },
        {"use_halves", 1, 0, USE_HALVES },
        {"astigmatism",1, 0, ASTIGMATISM},
        {"verbosity",  1, 0, VERBOSITY  },
        {"off_params", 1, 0, OFF_PARAM  },
        {0, 0, 0, 0}
    };
    
    while( (c=getopt_long_only(ac, av, "", long_options, 0)) >= 0 ) {
        switch(c) {
            case TOMOS_FILE:
                strcpy(info.tomo_file,optarg);
                break;
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
            case N_THREADS:
                info.n_threads = atoi(optarg);
                break;
            case GPU_LIST:
                info.n_gpu = ArgParser::get_list_integers(info.p_gpu,optarg);
                break;
            case PAD_SIZE:
                info.pad_size = ArgParser::get_even_number(optarg);
                break;
            case PAD_TYPE:
                info.pad_type = ArgParser::get_pad_type(optarg);
                break;
            case NORM_TYPE:
                info.norm_type = ArgParser::get_norm_type(optarg);
                break;
            case SSNR:
                ArgParser::get_single_pair(info.ssnr_F,info.ssnr_S,optarg);
                break;
            case BANDPASS:
                ArgParser::get_single_pair(info.fpix_min,info.fpix_max,optarg);
                break;
            case ROLLOFF_F:
                info.fpix_roll = atof(optarg);
                break;
            case DEF_SEARCH:
                ArgParser::get_single_pair(info.def_range,info.def_step,optarg);
                break;
            case ANG_SEARCH:
                ArgParser::get_single_pair(info.ang_range,info.ang_step,optarg);
                break;
            case EST_DOSE:
                info.est_dose = ArgParser::get_bool(optarg);
                break;
            case USE_HALVES:
                info.use_halves = ArgParser::get_bool(optarg);
                break;
            case ASTIGMATISM:
                info.astigmatism = ArgParser::get_bool(optarg);
                break;
            case VERBOSITY:
                info.verbosity = atoi(optarg);
                break;
            case OFF_PARAM:
                ArgParser::get_single_quad(info.off_x,info.off_y,info.off_z,info.off_s,optarg);
                break;
            default:
                printf("Unknown parameter %d\n",c);
                exit(1);
                break;
        } /// switch
    } /// while(c)

    return validate(info);
}

void print_full(const Info&info,FILE*fp=stdout) {
    fprintf(fp,"\tCTF Refiner");
    if( info.use_halves )
        fprintf(fp," (independent half-sets)");
    fprintf(fp,":\n");

    fprintf(fp,"\t\tParticles in:   %s.\n",info.ptcls_in);
    fprintf(fp,"\t\tTomograms file: %s.\n",info.tomo_file);
    fprintf(fp,"\t\tReference file: %s.\n",info.refs_file);
    fprintf(fp,"\t\tParticles out:  %s.\n",info.ptcls_out);

    fprintf(fp,"\t\tVolume size: %dx%dx%d",info.box_size,info.box_size,info.box_size);
    if( info.pad_size > 0 ) {
        fprintf(fp,", with padding of %d voxels",info.pad_size);
    }
    fprintf(fp,".\n");
    
    if( info.n_gpu > 1 ) {
        fprintf(fp,"\t\tUsing %d GPUs (GPU ids: %d",info.n_gpu,info.p_gpu[0]);
        for(int i=1;i<info.n_gpu;i++)
            fprintf(stdout,",%d",info.p_gpu[i]);
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

    fprintf(fp,"\t\tBandpass: [%.1f - %.1f] fourier pixels",info.fpix_min,info.fpix_max);
    if( info.fpix_roll > 0 )
        fprintf(fp," with a roll off of %.2f.\n",info.fpix_roll);
    else
        fprintf(fp,".\n");

    if( info.pad_size > 0 ) {
        if( info.pad_type == PAD_ZERO )
            fprintf(fp,"\t\tPadding policy: Fill with zeros.\n");
        if( info.pad_type == PAD_GAUSSIAN )
            fprintf(fp,"\t\tPadding policy: Fill with gaussian noise.\n");
    }
    
    if( info.norm_type == NO_NORM )
        fprintf(fp,"\t\tSubstack normalization policy: Disabled.\n");
    if( info.norm_type == ZERO_MEAN )
        fprintf(fp,"\t\tSubstack normalization policy: Mean=0.\n");
    if( info.norm_type == ZERO_MEAN_1_STD )
        fprintf(fp,"\t\tSubstack normalization policy: Mean=0, Std=1.\n");
    if( info.norm_type == ZERO_MEAN_W_STD )
        fprintf(fp,"\t\tSubstack normalization policy: Mean=0, Std according to projection weight.\n");

    fprintf(fp,"\t\tDefocus search range: %.2f.\n",info.def_range);
    fprintf(fp,"\t\tDefocus search step: %.2f.\n",info.def_step);
    fprintf(fp,"\t\tDefocus angle range: %.2f.\n",info.ang_range);
    fprintf(fp,"\t\tDefocus angle step: %.2f.\n",info.ang_step);

    if( info.est_dose )
        fprintf(fp,"\t\tWith dose weighting estimation.\n");
    else
        fprintf(fp,"\t\tWithout dose weighting estimation.\n");
}

void print_minimal(const Info&info,FILE*fp=stdout) {
    fprintf(fp,"  CTF Refiner. Box size: %d",info.box_size);
    if( info.pad_size > 0 ) {
        fprintf(fp," + %d (pad)",info.pad_size);
    }
    fprintf(fp,"\n");

    fprintf(fp,"    - Input files: %s | %s\n",info.tomo_file,info.refs_file);
    fprintf(fp,"    - Particles In/Out: %s | %s\n",info.ptcls_in,info.ptcls_out);

    if( info.n_gpu > 1 ) {
        fprintf(fp,"    - %d GPUs (GPU ids: %d",info.n_gpu,info.p_gpu[0]);
        for(int i=1;i<info.n_gpu;i++)
            fprintf(fp,",%d",info.p_gpu[i]);
        fprintf(fp,"), ");
    }
    else {
        fprintf(fp,"    - 1 GPU (GPU id: %d), ",info.p_gpu[0]);
    }

    if( info.n_threads > 1 ) {
        fprintf(fp,"and %d threads.\n",info.n_threads);
    }
    else{
        fprintf(fp,"and 1 thread.\n");
    }

    fprintf(fp,"    - Bandpass: [%.1f - %.1f] ",info.fpix_min,info.fpix_max);
    if( info.fpix_roll > 0 )
        fprintf(fp," (Smooth decay: %.2f).",info.fpix_roll);
    else
        fprintf(fp,".");
    
    if( info.norm_type == NO_NORM )
        fprintf(fp,"No Normalization.");
    if( info.norm_type == ZERO_MEAN )
        fprintf(fp,"Normalized to Mean=0.");
    if( info.norm_type == ZERO_MEAN_1_STD )
        fprintf(fp,"Normalized to Mean=0, Std=1.");
    if( info.norm_type == ZERO_MEAN_W_STD )
        fprintf(fp,"Normalized to Mean=0, Std=PRJ_W.");

    if( info.pad_size > 0 ) {
        if( info.pad_type == PAD_ZERO )
            fprintf(fp," Zero padding.\n");
        if( info.pad_type == PAD_GAUSSIAN )
            fprintf(fp," Random padding.\n");
    }
    else
        fprintf(fp,"\n");

    fprintf(fp,"    - Defocus search: %.3f,%.3f.",info.def_range,info.def_step);
    fprintf(fp,"\tDefocus angle %.3f,%.3f.\n",info.ang_range,info.ang_step);
}

void print(const Info&info,FILE*fp=stdout) {
    if( info.verbosity > 0 )
        print_full(info,fp);
    else
        print_minimal(info,fp);
}

}

#endif /// ALIGNER_ARGS_H

