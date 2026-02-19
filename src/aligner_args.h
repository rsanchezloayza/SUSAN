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

#ifndef ALIGNER_ARGS_H
#define ALIGNER_ARGS_H

#include <pthread.h>
#include <getopt.h>
#include "datatypes.h"
#include "arg_parser.h"
#include "gpu.h"
#include "angles_provider.h"
#include "math_cpu.h"
#include "points_provider.h"
#include "reference.h"

namespace ArgsAli {

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
    uint32 cc_type;
    uint32 cc_stats;
    uint32 type;
    uint32 dilate;
    float  ssnr_F;
    float  ssnr_S;
    bool   ali_halves;
    bool   ignore_ref;
    bool   drift;
    float  cone_range;
    float  cone_step;
    float  inplane_range;
    float  inplane_step;
    uint32 refine_level;
    uint32 refine_factor;
    uint32 off_type;
    uint32 off_space;
    float  off_x;
    float  off_y;
    float  off_z;
    float  off_s;

    char   pseudo_sym[64];

    char   refs_file[SUSAN_FILENAME_LENGTH];
    char   ptcls_out[SUSAN_FILENAME_LENGTH];
    char   ptcls_in [SUSAN_FILENAME_LENGTH];
    char   tomo_file[SUSAN_FILENAME_LENGTH];
    
    char   tm_type[SUSAN_FILENAME_LENGTH];
    char   tm_pfx [SUSAN_FILENAME_LENGTH];
    float  tm_sigma;
    
    int verbosity;
} Info;

bool validate(const Info&info) {
    bool rslt = true;
    if( (info.type < 2) || (info.type > 3) ) {
        fprintf(stderr,"Invalid type %d. Use 3 for 3D alignment or 2 for 2D.\n",info.type);
        rslt = false;
    }
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
        if( !refs.check_fields(info.ali_halves) ) {
            exit(1);
        }
        if( !refs.check_size(info.box_size,info.ali_halves) ) {
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
    if( !AnglesSymmetry::check_available_symmetry(info.pseudo_sym) ) {
        fprintf(stderr,"Invalid symmetry value: %s.\n",info.pseudo_sym);
        rslt = false;
    }
    if( !GPU::check_gpu_id_list(info.n_gpu,info.p_gpu) ) {
        fprintf(stderr,"Error with CUDA devices.\n");
        rslt = false;
    }
    if( !(strcmp(info.tm_type,"none") || strcmp(info.tm_type,"matlab") || strcmp(info.tm_type,"python") || strcmp(info.tm_type,"csv")) ) {
        fprintf(stderr,"Invalid template matching type [tm_type] value: %s. [none,matlab,python,csv].\n",info.tm_type);
        rslt = false;
    }
    return rslt;
}

bool parse_args(Info&info,int ac,char** av) {
    /// Default values:
    info.n_gpu         = 0;
    info.n_threads     = 1;
    info.box_size      = 200;
    info.fpix_min      = 0;
    info.fpix_max      = 30;
    info.fpix_roll     = 4;
    info.pad_size      = 0;
    info.pad_type      = PAD_ZERO;
    info.ctf_type      = ALI_CTF_ON_REFERENCE;
    info.norm_type     = NO_NORM;
    info.type          = 3;
    info.dilate        = 0;
    info.ssnr_F        = 0;
    info.ssnr_S        = 0;
    info.ali_halves    = false;
    info.ignore_ref    = false;
    info.cc_type       = CC_TYPE_BASIC;
    info.cc_stats      = CC_STATS_NONE;
    info.drift         = true;
    info.cone_range    = 0;
    info.cone_step     = 1;
    info.inplane_range = 0;
    info.inplane_step  = 1;
    info.refine_level  = 0;
    info.refine_factor = 1;
    info.off_type      = ELLIPSOID;
    info.off_space     = REFERENCE_SPACE;
    info.off_x         = 0;
    info.off_y         = 0;
    info.off_z         = 0;
    info.off_s         = 1;
    info.verbosity     = 0;
    memset(info.p_gpu    ,0,SUSAN_MAX_N_GPU*sizeof(uint32));
    memset(info.refs_file,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_out,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.ptcls_in ,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    memset(info.tomo_file,0,SUSAN_FILENAME_LENGTH*sizeof(char));
    strcpy(info.pseudo_sym,"c1");
    strcpy(info.tm_type,"none");
    strcpy(info.tm_pfx ,"template_matching");
    info.tm_sigma = 0;

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
        CTF_TYPE,
        CC_TYPE,
        CC_STATS,
        SSNR,
        BANDPASS,
        DILATE,
        ROLLOFF_F,
        SYMMETRY,
        ALI_HALVES,
        IGNORE_REF,
        DRIFT,
        CONE,
        INPLANE,
        REFINE,
        OFF_TYPE,
        OFF_SPACE,
        OFF_PARAM,
        VERBOSITY,
        TM_TYPE,
        TM_PREFIX,
        TM_SIGMA,
        TYPE
    };

    int c;
    static struct option long_options[] = {
        {"tomos_file",  1, 0, TOMOS_FILE},
        {"ptcls_in",    1, 0, PTCLS_IN  },
        {"ptcls_out",   1, 0, PTCLS_OUT },
        {"refs_file",   1, 0, REFS_FILE },
        {"n_threads",   1, 0, N_THREADS },
        {"gpu_list",    1, 0, GPU_LIST  },
        {"box_size",    1, 0, BOX_SIZE  },
        {"pad_size",    1, 0, PAD_SIZE  },
        {"pad_type",    1, 0, PAD_TYPE  },
        {"norm_type",   1, 0, NORM_TYPE },
        {"ctf_type",    1, 0, CTF_TYPE  },
        {"cc_type",     1, 0, CC_TYPE   },
        {"cc_stats",    1, 0, CC_STATS  },
        {"ssnr_param",  1, 0, SSNR      },
        {"bandpass",    1, 0, BANDPASS  },
        {"rolloff_f",   1, 0, ROLLOFF_F },
        {"p_symmetry",  1, 0, SYMMETRY  },
        {"ali_halves",  1, 0, ALI_HALVES},
        {"ignore_ref",  1, 0, IGNORE_REF},
        {"allow_drift", 1, 0, DRIFT     },
        {"cone",        1, 0, CONE      },
        {"inplane",     1, 0, INPLANE   },
        {"refine",      1, 0, REFINE    },
        {"off_type",    1, 0, OFF_TYPE  },
        {"off_space",   1, 0, OFF_SPACE },
        {"off_params",  1, 0, OFF_PARAM },
        {"verbosity",   1, 0, VERBOSITY },
        {"tm_type",     1, 0, TM_TYPE   },
        {"tm_prefix",   1, 0, TM_PREFIX },
        {"tm_sigma",    1, 0, TM_SIGMA  },
        {"dilate",      1, 0, DILATE    },
        {"type",        1, 0, TYPE },
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
            case CTF_TYPE:
                info.ctf_type = ArgParser::get_ali_ctf_type(optarg);
                /// ToBeDeprecated:
                if(strcmp(optarg,"cfsc")==0)
                    info.cc_type = CC_TYPE_CFSC;
                break;
            case CC_TYPE:
                info.cc_type = ArgParser::get_cc_type(optarg);
                break;
            case CC_STATS:
                info.cc_stats = ArgParser::get_cc_stats_type(optarg);
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
            case SYMMETRY:
                strcpy(info.pseudo_sym,optarg);
                break;
            case ALI_HALVES:
                info.ali_halves = ArgParser::get_bool(optarg);
                break;
            case IGNORE_REF:
                info.ignore_ref = ArgParser::get_bool(optarg);
                break;
            case DRIFT:
                info.drift = ArgParser::get_bool(optarg);
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
            case OFF_SPACE:
                info.off_space = ArgParser::get_offset_space(optarg);
                break;
            case OFF_PARAM:
                ArgParser::get_single_quad(info.off_x,info.off_y,info.off_z,info.off_s,optarg);
                break;
            case VERBOSITY:
                info.verbosity = atoi(optarg);
                break;
            case DILATE:
                info.dilate = atoi(optarg);
                break;
            case TYPE:
                info.type = atoi(optarg);
                break;
            case TM_TYPE:
                strcpy(info.tm_type,optarg);
                break;
            case TM_PREFIX:
                strcpy(info.tm_pfx,optarg);
                break;
            case TM_SIGMA:
                info.tm_sigma = atof(optarg);
                break;
            default:
                printf("Unknown parameter %d\n",c);
                exit(1);
                break;
        } /// switch
    } /// while(c)
    
    if( info.type == 2 ){
        if( (info.off_type == ELLIPSOID) || (info.off_type == CYLINDER) ){
            info.off_type = CIRCLE;
        }
        else if( info.off_type == CUBOID ){
            info.off_type = RECTANGLE;
        }
    }

    return validate(info);
}

void print_angles(const Info&info,FILE*fp,bool show_full=false) {
	
    AnglesProvider angles;
    angles.cone_range    = info.cone_range;
    angles.cone_step     = info.cone_step;
    angles.inplane_range = info.inplane_range;
    angles.inplane_step  = info.inplane_step;
    angles.refine_level  = info.refine_level;
    angles.refine_factor = info.refine_factor;
    angles.set_symmetry(info.pseudo_sym);

    int count_total=0;
    int count_level=0;
    int lvl = 0;

    if( !show_full ) fprintf(fp,"Angles:");
    for( angles.levels_init(); angles.levels_available(); angles.levels_next(), lvl++ ) {
        count_level=0;
        for( angles.sym_init(); angles.sym_available(); angles.sym_next() ) {
            for( angles.cone_init(); angles.cone_available(); angles.cone_next() ) {
                for( angles.inplane_init(); angles.inplane_available(); angles.inplane_next() ) {
                    count_level++;
                } /// inplane
            } /// cone
        } /// symmetry

        if( show_full )
            fprintf(fp,"\t\tAngles in refinement level %2d: %7d.\n",lvl,count_level);
        else {
            if( lvl > 0 ) fprintf(fp," |");
            fprintf(fp," %d",count_level);
        }
        count_total += count_level;
    } /// level
    if( show_full )
        fprintf(fp,"\t\tTotal angles: %d.\n",count_total);
    else
        fprintf(fp," [Total: %d].\n",count_total);
}

void print_full(const Info&info,FILE*fp) {
    fprintf(fp,"\tVolume %dD alignment",info.type);
    if( info.ali_halves )
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

    if( info.ctf_type == ALI_CTF_DISABLED )
        fprintf(fp,"\t\tCTF correction policy: Disabled.\n");
    if( info.ctf_type == ALI_CTF_PHASE_FLIP)
        fprintf(fp,"\t\tCTF correction policy: Phase-flip.\n");
    if( info.ctf_type == ALI_CTF_ON_REFERENCE )
        fprintf(fp,"\t\tCTF correction policy: On reference.\n");
    if( info.ctf_type == ALI_CTF_ON_SUBSTACK )
        fprintf(fp,"\t\tCTF correction policy: On substack - Wiener inversion.\n");
    if( info.ctf_type == ALI_CTF_ON_SUBSTACK_SSNR )
        fprintf(fp,"\t\tCTF correction policy: On substack - Wiener inversion with SSNR(f) = (10^(3*%.2f))*e^(-100*%.2f*f).\n",info.ssnr_S,info.ssnr_F);

    if( info.cc_type == CC_TYPE_BASIC )
        fprintf(fp,"\t\tAligning using Cross-Correlation. ");
    if( info.cc_type == CC_TYPE_CFSC )
        fprintf(fp,"\t\tAligning using the Cumulative Fourier Shell Correlation. ");

    if( info.cc_stats == CC_STATS_NONE )
        fprintf(fp,"\n");
    if( info.cc_stats == CC_STATS_SIGMA )
        fprintf(fp,"Measuring: max( (cc_max - cc_mean) / cc_std , 0 ) per angle.\n");
    if( info.cc_stats == CC_STATS_PROB )
        fprintf(fp,"Measuring: cc_max / sum( cc ).\n");

    if( info.norm_type == NO_NORM )
        fprintf(fp,"\t\tSubstack normalization policy: Disabled.\n");
    if( info.norm_type == ZERO_MEAN )
        fprintf(fp,"\t\tSubstack normalization policy: Mean=0.\n");
    if( info.norm_type == ZERO_MEAN_1_STD )
        fprintf(fp,"\t\tSubstack normalization policy: Mean=0, Std=1.\n");
    if( info.norm_type == ZERO_MEAN_W_STD )
        fprintf(fp,"\t\tSubstack normalization policy: Mean=0, Std according to projection weight.\n");
    if( info.norm_type == GAT_RAW )
        fprintf(fp,"\t\tSubstack normalization policy: VST.\n");
    if( info.norm_type == GAT_NORMAL )
        fprintf(fp,"\t\tSubstack normalization policy: GAT.\n");

    fprintf(fp,"\t\tPseudo-symmetry search: %s.\n",info.pseudo_sym);
    fprintf(fp,"\t\tCone search:    Range=%.3f, Step=%.3f.\n",info.cone_range,info.cone_step);
    fprintf(fp,"\t\tInplane search: Range=%.3f, Step=%.3f.\n",info.inplane_range,info.inplane_step);
    fprintf(fp,"\t\tAngle refinement: Levels=%d, Factor=%d.\n",info.refine_level,info.refine_factor);
    print_angles(info,fp,info.verbosity>0);
    
    uint32_t total_points=0;
    if( info.off_type == ELLIPSOID ) {
        Vec3*pt = PointsProvider::ellipsoid(total_points,info.off_x,info.off_y,info.off_z,info.off_s);
        fprintf(fp,"\t\tEllipsoid offset search (3D): ");
        delete [] pt;
        fprintf(fp,"Range=[%.2f,%.2f,%.2f], Step=%.2f. Total points: %d\n",info.off_x,info.off_y,info.off_z,info.off_s,total_points);
    }
    if( info.off_type == CYLINDER ) {
        Vec3*pt = PointsProvider::cylinder(total_points,info.off_x,info.off_y,info.off_z,info.off_s);
        fprintf(fp,"\t\tCylindrical offset search (3D): ");
        delete [] pt;
        fprintf(fp,"Range=[%.2f,%.2f,%.2f], Step=%.2f. Total points: %d\n",info.off_x,info.off_y,info.off_z,info.off_s,total_points);
    }
    if( info.off_type == CUBOID ) {
        Vec3*pt = PointsProvider::cuboid(total_points,info.off_x,info.off_y,info.off_z,info.off_s);
        fprintf(fp,"\t\tCuboid offset search (3D): ");
        delete [] pt;
        fprintf(fp,"Range=[%.2f,%.2f,%.2f], Step=%.2f. Total points: %d\n",info.off_x,info.off_y,info.off_z,info.off_s,total_points);
    }
    if( info.off_type == CIRCLE ) {
        Vec3*pt = PointsProvider::circle(total_points,info.off_x,info.off_y,info.off_s);
        fprintf(fp,"\t\tCircular offset search (2D): ");
        delete [] pt;
        fprintf(fp,"Range=[%.2f,%.2f]. Total points: %d\n",info.off_x,info.off_y,total_points);
    }
    if( info.off_type == RECTANGLE ) {
        Vec3*pt = PointsProvider::rectangle(total_points,info.off_x,info.off_y,info.off_s);
        fprintf(fp,"\t\tRectangular offset search (2D): ");
        delete [] pt;
        fprintf(fp,"Range=[%.2f,%.2f]. Total points: %d\n",info.off_x,info.off_y,total_points);
    }
    
    if( strcmp(info.tm_type,"none") != 0 ) {
        fprintf(fp,"\t\tSaving Cross Correlation in %s format. Prefix: %s.",info.tm_type,info.tm_pfx);
        if( info.tm_sigma > 0 ) {
            fprintf(fp," Discard CC < %.1fσ.",info.tm_sigma);
        }
        fprintf(fp,"\n");
    }

    if( info.ignore_ref )
        fprintf(fp,"\t\tAligning each particle to its class/reference only.\n");
}

void print_minimal(const Info&info,FILE*fp) {
    fprintf(fp,"  Volume %dD alignment",info.type);
    if( info.ali_halves )
        fprintf(fp," (independent half-sets)");
    
    fprintf(fp,". Box size: %d",info.box_size);
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
        fprintf(fp," (Smooth decay: %.2f).\n",info.fpix_roll);
    else
        fprintf(fp,".\n");

    fprintf(fp,"    - ");
    if( info.pad_size > 0 ) {
        if( info.pad_type == PAD_ZERO )
            fprintf(fp,"Zero padding. ");
        if( info.pad_type == PAD_GAUSSIAN )
            fprintf(fp,"Random padding. ");
    }

    if( info.ctf_type == ALI_CTF_DISABLED )
        fprintf(fp,"No CTF correction. ");
    if( info.ctf_type == ALI_CTF_PHASE_FLIP)
        fprintf(fp,"CTF by phase-flip. ");
    if( info.ctf_type == ALI_CTF_ON_REFERENCE )
        fprintf(fp,"CTF on Reference. ");
    if( info.ctf_type == ALI_CTF_ON_SUBSTACK )
        fprintf(fp,"CTF on Substack (Wiener inversion). ");
    if( info.ctf_type == ALI_CTF_ON_SUBSTACK_SSNR )
        fprintf(fp,"CTF on Substack (Wiener inversion with Ad-Hoc SSNR: S=%.2f F=%.2f). ",info.ssnr_S,info.ssnr_F);

    if( info.norm_type == NO_NORM )
        fprintf(fp,"No Normalization.\n");
    if( info.norm_type == ZERO_MEAN )
        fprintf(fp,"Normalized to Mean=0.\n");
    if( info.norm_type == ZERO_MEAN_1_STD )
        fprintf(fp,"Normalized to Mean=0, Std=1.\n");
    if( info.norm_type == ZERO_MEAN_W_STD )
        fprintf(fp,"Normalized to Mean=0, Std=PRJ_W.\n");
    if( info.norm_type == GAT_RAW )
        fprintf(fp,"VST Normalization.\n");
    if( info.norm_type == GAT_NORMAL )
        fprintf(fp,"GAT Normalization.\n");
    
    fprintf(fp,"    - ");
    if( info.cc_type == CC_TYPE_BASIC )
        fprintf(fp,"Aligning with CC. ");
    if( info.cc_type == CC_TYPE_CFSC )
        fprintf(fp,"Aligning with CFSC. ");

    if( info.cc_stats == CC_STATS_SIGMA )
        fprintf(fp,"Measuring (per angle): max( (cc_max - cc_mean) / cc_std , 0 ).");
    if( info.cc_stats == CC_STATS_PROB )
        fprintf(fp,"Measuring: cc_max / sum(cc).");
    fprintf(fp,"\n");

    fprintf(fp,"    - Angular search: [ %s | ",info.pseudo_sym);
    
    fprintf(fp,"%.3f,%.3f | ",info.cone_range,info.cone_step);
    fprintf(fp,"%.3f,%.3f | ",info.inplane_range,info.inplane_step);
    fprintf(fp,"%d|%d ]: ",info.refine_level,info.refine_factor);
    print_angles(info,fp,info.verbosity>0);
	
    uint32_t total_points=0;
    if( info.off_type == ELLIPSOID ) {
        Vec3*pt = PointsProvider::ellipsoid(total_points,info.off_x,info.off_y,info.off_z,info.off_s);
        fprintf(fp,"    - Ellipsoid offset (3D): ");
        delete [] pt;
        fprintf(fp,"[%.2f,%.2f,%.2f], Step=%.2f. Points: %d\n",info.off_x,info.off_y,info.off_z,info.off_s,total_points);
    }
    if( info.off_type == CYLINDER ) {
        Vec3*pt = PointsProvider::cylinder(total_points,info.off_x,info.off_y,info.off_z,info.off_s);
        fprintf(fp,"    - Cylindrical offset (3D): ");
        delete [] pt;
        fprintf(fp,"[%.2f,%.2f,%.2f], Step=%.2f. Points: %d\n",info.off_x,info.off_y,info.off_z,info.off_s,total_points);
    }
    if( info.off_type == CUBOID ) {
        Vec3*pt = PointsProvider::cuboid(total_points,info.off_x,info.off_y,info.off_z,info.off_s);
        fprintf(fp,"    - Cuboid offset (3D): ");
        delete [] pt;
        fprintf(fp,"[%.2f,%.2f,%.2f], Step=%.2f. Points: %d\n",info.off_x,info.off_y,info.off_z,info.off_s,total_points);
    }
    if( info.off_type == CIRCLE ) {
        Vec3*pt = PointsProvider::circle(total_points,info.off_x,info.off_y,info.off_s);
        fprintf(fp,"    - Circular offset (2D): ");
        delete [] pt;
        fprintf(fp,"[%.2f,%.2f]. Points: %d\n",info.off_x,info.off_y,total_points);
    }
    if( info.off_type == RECTANGLE ) {
        Vec3*pt = PointsProvider::rectangle(total_points,info.off_x,info.off_y,info.off_s);
        fprintf(fp,"    - Rectangular offset (2D): ");
        delete [] pt;
        fprintf(fp,"[%.2f,%.2f]. Points: %d\n",info.off_x,info.off_y,total_points);
    }
    
    if( strcmp(info.tm_type,"none") != 0 ) {
        fprintf(fp,"    - Saving Cross Correlation in %s format. Prefix: %s.",info.tm_type,info.tm_pfx);
        if( info.tm_sigma > 0 ) {
            fprintf(fp," Discard CC < %.1fσ.",info.tm_sigma);
        }
        fprintf(fp,"\n");
    }

    if( info.ignore_ref )
        fprintf(fp,"    - Aligning each particle to its class/reference only.\n");
}

void print(const Info&info,FILE*fp=stdout) {
	if( info.verbosity > 0 )
		print_full(info,fp);
	else
		print_minimal(info,fp);
}

}

#endif /// ALIGNER_ARGS_H

