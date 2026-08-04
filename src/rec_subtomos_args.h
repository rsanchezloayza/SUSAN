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

#ifndef REC_SUBTOMOS_ARGS_H
#define REC_SUBTOMOS_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "gpu.h"
#include "io.h"
#include "arg_parser.h"

namespace ArgsRecSubtomo {

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
    uint32 norm_type;
    uint32 out_fmt;
    bool   use_ali;
    bool   invert;
    bool   relion_ctf;
    int    w_inv_ite;
    float  w_inv_std;
    float  ssnr_F;
    float  ssnr_S;
    bool   norm_output;
    float  boost_low_fq_scale;
    float  boost_low_fq_value;
    float  boost_low_fq_decay;

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
    if( !GPU::check_gpu_id_list(info.n_gpu,info.p_gpu) ) {
        fprintf(stderr,"Error with CUDA devices.\n");
        rslt = false;
    }
    return rslt;
};

inline bool parse_args(Info&info,int ac,char** av) {
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
    info.norm_type   = NO_NORM;
    info.out_fmt     = CROP_MRC;
    info.w_inv_ite   = 10;
    info.w_inv_std   = -1;
    info.use_ali     = false;
    info.relion_ctf  = false;
    info.norm_output = true;
    info.ssnr_F      = 0;
    info.invert      = false;
    info.ssnr_S      = 1;
    memset(info.p_gpu   ,0,SUSAN_MAX_N_GPU*sizeof(uint32));
    memset(info.out_dir ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_in,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.tomos_in,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    info.boost_low_fq_scale = 0;
    info.boost_low_fq_value = 0;
    info.boost_low_fq_decay = 0;

    /// Parse inputs:
    enum {
        TOMOS_FILE,
        OUT_FOLDER,
        PTCLS_FILE,
        N_THREADS,
        GPU_LIST,
        BOX_SIZE,
        PAD_SIZE,
        PAD_TYPE,
        NORM_TYPE,
        FORMAT,
        CTF_TYPE,
        SSNR,
        W_INV_ITE,
        W_INV_STD,
        BANDPASS,
        ROLLOFF_F,
        USE_ALI,
        INVERT,
        RELION_CTF,
        NORM_OUTPUT,
        BOOST_LOWFQ
    };

    int c;
    static struct option long_options[] = {
        {"tomos_file",  1, 0, TOMOS_FILE },
        {"out_dir",     1, 0, OUT_FOLDER },
        {"ptcls_file",  1, 0, PTCLS_FILE },
        {"n_threads",   1, 0, N_THREADS  },
        {"gpu_list",    1, 0, GPU_LIST   },
        {"box_size",    1, 0, BOX_SIZE   },
        {"pad_size",    1, 0, PAD_SIZE   },
        {"pad_type",    1, 0, PAD_TYPE   },
        {"norm_type",   1, 0, NORM_TYPE  },
        {"format",      1, 0, FORMAT     },
        {"ctf_type",    1, 0, CTF_TYPE   },
        {"ssnr_param",  1, 0, SSNR       },
        {"w_inv_iter",  1, 0, W_INV_ITE  },
        {"w_inv_gstd",  1, 0, W_INV_STD  },
        {"use_align",   1, 0, USE_ALI    },
        {"relion_ctf",  1, 0, RELION_CTF },
        {"invert",      1, 0, INVERT     },
        {"norm_output", 1, 0, NORM_OUTPUT},
        {"boost_lowfq", 1, 0, BOOST_LOWFQ},
        {"bandpass",    1, 0, BANDPASS   },
        {"rolloff_f",   1, 0, ROLLOFF_F  },
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
            case FORMAT:
                info.out_fmt = ArgParser::get_format_output(optarg);
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
            case USE_ALI:
                info.use_ali = ArgParser::get_bool(optarg);
                break;
            case INVERT:
                info.invert = ArgParser::get_bool(optarg);
                break;
            case RELION_CTF:
                info.relion_ctf = ArgParser::get_bool(optarg);
                break;
            case NORM_OUTPUT:
                info.norm_output = ArgParser::get_bool(optarg);
                break;
            case BOOST_LOWFQ:
                ArgParser::get_single_trio(info.boost_low_fq_scale,info.boost_low_fq_value,info.boost_low_fq_decay,optarg);
                break;
            default:
                printf("Unknown parameter %d\n",c);
                exit(1);
                break;
        } /// switch
    } /// while(c)

    if( info.w_inv_std <= 0 )
        info.w_inv_std = W_INV_STD_LINEAR;

    return validate(info);
}

inline void print(const Info&info,FILE*fp=stdout) {
    fprintf(stdout,"\tSubtomogram generation:\n");

    fprintf(stdout,"\t\tParticles file: %s.\n",info.ptcls_in);
    fprintf(stdout,"\t\tTomograms file: %s.\n",info.tomos_in);
    fprintf(stdout,"\t\tOutput folder: %s.\n",info.out_dir);

    fprintf(stdout,"\t\tVolume size: %dx%dx%d",info.box_size,info.box_size,info.box_size);
    if( info.pad_size > 0 ) {
        fprintf(stdout,", with padding of %d voxels",info.pad_size);
    }
    fprintf(stdout,".\n");

    if( info.use_ali ) {
        fprintf(stdout,"\t\tReconstructing aligned subtomograms.");
    }
    else{
        fprintf(stdout,"\t\tReconstructing raw subtomograms (ignoring alignment information).");
    }
    fprintf(stdout,".\n");

    if( info.n_gpu > 1 ) {
        fprintf(stdout,"\t\tUsing %d GPUs (GPU ids: %d",info.n_gpu,info.p_gpu[0]);
        for(int i=1;i<info.n_gpu;i++)
            fprintf(stdout,",%d",info.p_gpu[i]);
        fprintf(stdout,"), ");
    }
    else {
        fprintf(stdout,"\t\tUsing 1 GPU (GPU id: %d), ",info.p_gpu[0]);
    }

    if( info.n_threads > 1 ) {
        fprintf(stdout,"and %d threads.\n",info.n_threads);
    }
    else{
        fprintf(stdout,"and 1 thread.\n");
    }

    if( info.pad_size > 0 ) {
        if( info.pad_type == PAD_ZERO )
            fprintf(stdout,"\t\tPadding policy: Fill with zeros.\n");
        if( info.pad_type == PAD_GAUSSIAN )
            fprintf(stdout,"\t\tPadding policy: Fill with gaussian noise.\n");
    }

    if( info.ctf_type == INV_NO_INV )
        fprintf(stdout,"\t\tCTF correction policy: Disabled.\n");
    if( info.ctf_type == INV_PHASE_FLIP )
        fprintf(stdout,"\t\tCTF correction policy: Phase-flip.\n");
    if( info.ctf_type == INV_WIENER )
        fprintf(stdout,"\t\tCTF correction policy: Wiener inversion.\n");
    if( info.ctf_type == INV_WIENER_SSNR )
        fprintf(stdout,"\t\tCTF correction policy: Wiener inversion with SSNR(f) = (100^(3*%.2f))*e^(-100*%.2f*f).\n",info.ssnr_S,info.ssnr_F);

    if( info.w_inv_ite > 0 )
        fprintf(stdout,"\t\tIterative inversion of the fourier weights (Ite=%d, Std=%f).\n",info.w_inv_ite,info.w_inv_std);
    else
        fprintf(stdout,"\t\tDirect inversion of the fourier weights.\n");

    if( info.norm_type == NO_NORM )
        fprintf(stdout,"\t\tSubstack normalization policy: Disabled.\n");
    if( info.norm_type == ZERO_MEAN )
        fprintf(stdout,"\t\tSubstack normalization policy: Mean=0.\n");
    if( info.norm_type == ZERO_MEAN_1_STD )
        fprintf(stdout,"\t\tSubstack normalization policy: Mean=0, Std=1.\n");
    if( info.norm_type == ZERO_MEAN_W_STD )
        fprintf(stdout,"\t\tSubstack normalization policy: Mean=0, Std according to projection weight.\n");

    if( info.norm_output )
        fprintf(fp,"\t\tNormalizing output.\n");
    else
        fprintf(fp,"\t\tDo not normalizing output.\n");

    if( info.boost_low_fq_scale > 0 )
        fprintf(fp,"\t\tBoosting low frequencies (Scale=%.2f, FPix=%.1f, Decay=%.1f).\n",info.boost_low_fq_scale,info.boost_low_fq_value,info.boost_low_fq_decay);

}


}

#endif /// REC_SUBTOMOS_ARGS_H

