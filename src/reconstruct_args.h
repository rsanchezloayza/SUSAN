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

#ifndef RECONSTRUCT_ARGS_H
#define RECONSTRUCT_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "arg_parser.h"
#include "io.h"
#include "gpu.h"
#include "angles_symmetry.h"

namespace ArgsRec {

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
    uint32 ctf_type;
    uint32 wgt_type;
    uint32 norm_type;
    uint32 grid_type;
    int    w_inv_ite;
    float  w_inv_std;
    float  ssnr_F;
    float  ssnr_S;
    bool   rec_halves;
    bool   ignore_ref;
    bool   norm_output;
    float  boost_low_fq_scale;
    float  boost_low_fq_value;
    float  boost_low_fq_decay;

    char   sym[64];

    char   out_pfx[SUSAN_FILENAME_LENGTH];
    char   ptcls_in[SUSAN_FILENAME_LENGTH];
    char   tomos_in[SUSAN_FILENAME_LENGTH];

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
    if( !IO::exists(info.tomos_in) ) {
        fprintf(stderr,"Tomos file %s does not exist.\n",info.tomos_in);
        rslt = false;
    }
    if( strlen(info.out_pfx) == 0 ) {
        fprintf(stderr,"Output pfx missing.\n");
        rslt = false;
    }
    if( !AnglesSymmetry::check_available_symmetry(info.sym) ) {
        fprintf(stderr,"Invalid symmetry value: %s.\n",info.sym);
        rslt = false;
    }
    if( !GPU::check_gpu_id_list(info.n_gpu,info.p_gpu) ) {
        fprintf(stderr,"Error with CUDA devices.\n");
        rslt = false;
    }
    return rslt;
};

bool parse_args(Info&info,int ac,char** av) {
    /// Default values:
    info.n_gpu       = 0;
    info.n_threads   = 1;
    info.box_size    = 200;
    info.fpix_min    = 0;
    info.fpix_max    = 30;
    info.fpix_roll   = 4;
    info.pad_size    = 0;
    info.pad_type    = PAD_ZERO;
    info.ctf_type    = INV_WIENER;
    info.wgt_type    = WGT_NONE;
    info.norm_type   = NO_NORM;
    info.grid_type   = GRIDDING_LINEAR_FWD;
    info.w_inv_ite   = 10;
    info.w_inv_std   = 0.75;
    info.ssnr_F      = 0;
    info.ssnr_S      = 1;
    info.rec_halves  = false;
    info.ignore_ref  = false;
    info.norm_output = true;
    info.verbosity   = 0;
    memset(info.p_gpu   ,0,SUSAN_MAX_N_GPU*sizeof(uint32));
    memset(info.out_pfx ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_in,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.tomos_in,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    strcpy(info.sym,"c1");
    info.boost_low_fq_scale = 0;
    info.boost_low_fq_value = 0;
    info.boost_low_fq_decay = 0;

    /// Parse inputs:
    enum {
        TOMOS_FILE,
        OUT_PREFIX,
        PTCLS_FILE,
        N_THREADS,
        GPU_LIST,
        BOX_SIZE,
        PAD_SIZE,
        PAD_TYPE,
        NORM_TYPE,
        CTF_TYPE,
        WGT_TYPE,
        GRID_TYPE,
        SSNR,
        W_INV_ITE,
        W_INV_STD,
        BANDPASS,
        ROLLOFF_F,
        SYMMETRY,
        VERBOSITY,
        REC_HALVES,
        IGNORE_REF,
        NORM_OUTPUT,
        BOOST_LOWFQ
    };

    int c;
    static struct option long_options[] = {
        {"tomos_file",  1, 0, TOMOS_FILE },
        {"out_prefix",  1, 0, OUT_PREFIX },
        {"ptcls_file",  1, 0, PTCLS_FILE },
        {"n_threads",   1, 0, N_THREADS  },
        {"gpu_list",    1, 0, GPU_LIST   },
        {"box_size",    1, 0, BOX_SIZE   },
        {"pad_size",    1, 0, PAD_SIZE   },
        {"pad_type",    1, 0, PAD_TYPE   },
        {"norm_type",   1, 0, NORM_TYPE  },
        {"ctf_type",    1, 0, CTF_TYPE   },
        {"wgt_type",    1, 0, WGT_TYPE   },
        {"grid_type",   1, 0, GRID_TYPE  },
        {"ssnr_param",  1, 0, SSNR       },
        {"w_inv_iter",  1, 0, W_INV_ITE  },
        {"w_inv_gstd",  1, 0, W_INV_STD  },
        {"bandpass",    1, 0, BANDPASS   },
        {"rolloff_f",   1, 0, ROLLOFF_F  },
        {"symmetry",    1, 0, SYMMETRY   },
        {"rec_halves",  1, 0, REC_HALVES },
        {"ignore_ref",  1, 0, IGNORE_REF },
        {"norm_output", 1, 0, NORM_OUTPUT},
        {"boost_lowfq", 1, 0, BOOST_LOWFQ},
        {"verbosity",   1, 0, VERBOSITY  },
        {0, 0, 0, 0}
    };

    while( (c=getopt_long_only(ac, av, "", long_options, 0)) >= 0 ) {
        switch(c) {
            case TOMOS_FILE:
                strcpy(info.tomos_in,optarg);
                break;
            case OUT_PREFIX:
                strcpy(info.out_pfx,optarg);
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
            case CTF_TYPE:
                info.ctf_type = ArgParser::get_inv_ctf_type(optarg);
                break;
            case WGT_TYPE:
                info.wgt_type = ArgParser::get_weighting_type(optarg);
                break;
            case GRID_TYPE:
                info.grid_type = ArgParser::get_gridding_type(optarg);
                break;
            case BANDPASS:
                ArgParser::get_single_pair(info.fpix_min,info.fpix_max,optarg);
                break;
            case ROLLOFF_F:
                info.fpix_roll = atof(optarg);
                break;
            case SSNR:
                ArgParser::get_single_pair(info.ssnr_F,info.ssnr_S,optarg);
                break;
            case W_INV_ITE:
                info.w_inv_ite = atoi(optarg);
                break;
            case W_INV_STD:
                info.w_inv_std = atof(optarg);
                break;
            case SYMMETRY:
                strcpy(info.sym,optarg);
                break;
            case REC_HALVES:
                info.rec_halves = ArgParser::get_bool(optarg);
                break;
            case IGNORE_REF:
                info.ignore_ref = ArgParser::get_bool(optarg);
                break;
            case NORM_OUTPUT:
                info.norm_output = ArgParser::get_bool(optarg);
                break;
            case BOOST_LOWFQ:
                ArgParser::get_single_trio(info.boost_low_fq_scale,info.boost_low_fq_value,info.boost_low_fq_decay,optarg);
                break;
            case VERBOSITY:
                info.verbosity = atoi(optarg);
                break;
            default:
                printf("Unknown parameter %d\n",c);
                exit(1);
                break;
        } /// switch
    } /// while(c)

    return validate(info);
}

void print_full(const Info&info,FILE*fp) {
    fprintf(fp,"\tVolume reconstruction");
    if( info.rec_halves )
        fprintf(fp," (including half-sets)");
    fprintf(fp,":\n");

    fprintf(fp,"\t\tParticles file: %s.\n",info.ptcls_in);
    fprintf(fp,"\t\tTomograms file: %s.\n",info.tomos_in);
    fprintf(fp,"\t\tOutput prefix: %s.\n",info.out_pfx);

    fprintf(fp,"\t\tVolume size: %dx%dx%d",info.box_size,info.box_size,info.box_size);
    if( info.pad_size > 0 ) {
        fprintf(fp,", with padding of %d voxels",info.pad_size);
    }
    fprintf(fp,".\n");

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

    if( info.ctf_type == INV_NO_INV )
        fprintf(fp,"\t\tCTF correction policy: Disabled.\n");
    if( info.ctf_type == INV_PHASE_FLIP )
        fprintf(fp,"\t\tCTF correction policy: Phase-flip.\n");
    if( info.ctf_type == INV_WIENER )
        fprintf(fp,"\t\tCTF correction policy: Wiener inversion.\n");
    if( info.ctf_type == INV_WIENER_SSNR )
        fprintf(fp,"\t\tCTF correction policy: Wiener inversion with SSNR(f) = (100^(3*%.2f))*e^(-100*%.2f*f).\n",info.ssnr_S,info.ssnr_F);

    fprintf(fp,"\t\tInversion of the sampled fourier space using %d iterations and a gaussian filter with std of %f.\n",info.w_inv_ite,info.w_inv_std);

    if( info.norm_type == NO_NORM )
        fprintf(fp,"\t\tSubstack normalization policy: Disabled.\n");
    if( info.norm_type == ZERO_MEAN )
        fprintf(fp,"\t\tSubstack normalization policy: Mean=0.\n");
    if( info.norm_type == ZERO_MEAN_1_STD )
        fprintf(fp,"\t\tSubstack normalization policy: Mean=0, Std=1.\n");
    if( info.norm_type == ZERO_MEAN_W_STD )
        fprintf(fp,"\t\tSubstack normalization policy: Mean=0, Std according to projection weight.\n");

    fprintf(fp,"\t\tSymmetry type: %s.\n",info.sym);
    if( !info.norm_output )
        fprintf(fp,"\t\tDo not normalize output.\n");

    if( info.boost_low_fq_scale > 0 )
        fprintf(fp,"\t\tBoosting low frequencies (Scale=%.2f, FPix=%.1f, Decay=%.1f).\n",info.boost_low_fq_scale,info.boost_low_fq_value,info.boost_low_fq_decay);

    if( info.ignore_ref )
        fprintf(fp,"\t\tIgnoring class information. All particles contribute to all classes.\n");

}

void print_basic(const Info&info,FILE*fp) {
    fprintf(fp,"    Volume reconstruction");
    if( info.rec_halves )
        fprintf(fp," (including half-sets)");

    fprintf(fp,". Box size: %d",info.box_size);
    if( info.pad_size > 0 ) {
        fprintf(fp," + %d (pad)",info.pad_size);
    }
    fprintf(fp,"\n");

    fprintf(fp,"    - [ %s | %s ]",info.ptcls_in,info.tomos_in);
    fprintf(fp," -> [ Prefix: %s ]\n",info.out_pfx);

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
        fprintf(fp," %d threads.\n",info.n_threads);
    }
    else{
        fprintf(fp," 1 thread.\n");
    }

    fprintf(fp,"    - Bandpass: [%.1f - %.1f]",info.fpix_min,info.fpix_max);
    if( info.fpix_roll > 0 )
        fprintf(fp," (Smooth decay: %.2f)",info.fpix_roll);

    fprintf(fp,". %s Symmetry.\n",info.sym);

    fprintf(fp,"    - ");
    if( info.pad_size > 0 ) {
        if( info.pad_type == PAD_ZERO )
            fprintf(fp,"Zero padding. ");
        if( info.pad_type == PAD_GAUSSIAN )
            fprintf(fp,"Random padding. ");
    }

    if( info.ctf_type == INV_NO_INV )
        fprintf(fp,"No CTF inversion. ");
    if( info.ctf_type == INV_PHASE_FLIP )
        fprintf(fp,"Phase-flip correction. ");
    if( info.ctf_type == INV_WIENER )
        fprintf(fp,"Wiener Inversion. ");
    if( info.ctf_type == INV_WIENER_SSNR )
        fprintf(fp,"Wiener SSNR (S=%.2f F=%.2f). ",info.ssnr_S,info.ssnr_F);

    if( info.norm_type == NO_NORM )
        fprintf(fp,"No Normalization.\n");
    if( info.norm_type == ZERO_MEAN )
        fprintf(fp,"Normalization (Mean=0).\n");
    if( info.norm_type == ZERO_MEAN_1_STD )
        fprintf(fp,"Normalization (Mean=0, Std=1).\n");
    if( info.norm_type == ZERO_MEAN_W_STD )
        fprintf(fp,"Normalization (Mean=0, Std=PRJ_W).\n");

    if( !info.norm_output || info.boost_low_fq_scale > 0 || info.ignore_ref ) {
        fprintf(fp,"    - ");

        if( !info.norm_output )
            fprintf(fp,"Do not normalize output. ");

        if( info.boost_low_fq_scale > 0 )
            fprintf(fp,"Boosting low frequencies (Scale=%.2f, FPix=%.1f, Decay=%.1f). ",info.boost_low_fq_scale,info.boost_low_fq_value,info.boost_low_fq_decay);

        if( info.ignore_ref )
            fprintf(fp,"All particles to all classes. ");

        fprintf(fp,"\n");
    }
}

void print_minimal(const Info&info,FILE*fp) {
    if( info.rec_halves )
        fprintf(fp,"    Reconstructing halfmaps");
    else
        fprintf(fp,"    Reconstructing volumes");

    fprintf(fp," [%d",info.box_size);
    if( info.pad_size > 0 )
        fprintf(fp,"+%d",info.pad_size);
    fprintf(fp,"]:");

    fprintf(fp," %s | %s ",info.ptcls_in,info.tomos_in);
    fprintf(fp,"-> %s",info.out_pfx);

    if( info.n_gpu > 1 )
        fprintf(fp," on %d GPUs.\n",info.n_gpu);
    else
        fprintf(fp," on 1 GPU.\n");
}

void print(const Info&info,FILE*fp=stdout) {
    if( info.verbosity == VERBOSITY_FULL )
        print_full(info,fp);
    else if( info.verbosity == VERBOSITY_BASIC )
        print_basic(info,fp);
    else
        print_minimal(info,fp);
}


}

#endif /// RECONSTRUCT_ARGS_H

