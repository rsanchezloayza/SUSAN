/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2022 Ricardo Miguel Sanchez Loayza.
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

#pragma once

#include "datatypes.h"

namespace ArgParser {

    template<typename T>
    void check_arg_and_set(T&value,bool&ok,const char*arg,const char*arg_name,const T arg_value) {
        if( strcmp(arg,arg_name) == 0 ) {
            value = arg_value;
            ok = true;
        }
    }

    PaddingType_t get_pad_type(const char*arg) {
        PaddingType_t type = PAD_ZERO;
        bool all_ok = false;

        check_arg_and_set(type,all_ok,arg,"zero" ,PAD_ZERO    );
        check_arg_and_set(type,all_ok,arg,"noise",PAD_GAUSSIAN);

        if( !all_ok )
            fprintf(stderr,"Invalid padding type %s. Options are: zero or noise. Defaulting to zero.\n",arg);

        return type;
    }

    CropFormat_t get_format_output(const char*arg) {
        CropFormat_t type = CROP_MRC;
        bool all_ok = false;

        check_arg_and_set(type,all_ok,arg,"mrc",CROP_MRC);
        check_arg_and_set(type,all_ok,arg,"em" ,CROP_EM);

        if( !all_ok )
            fprintf(stderr,"Invalid format type %s. Options are: mrc or em. Defaulting to mrc.\n",arg);

        return type;
    }

    CcType_t get_cc_type(const char*arg) {
        CcType_t type = CC_TYPE_BASIC;
        bool all_ok = false;

        check_arg_and_set(type,all_ok,arg,"basic",CC_TYPE_BASIC);
        check_arg_and_set(type,all_ok,arg,"cfsc" ,CC_TYPE_CFSC );

        if( !all_ok )
            fprintf(stderr,"Invalid cc type %s. Options are: basic or cfsc. Defaulting to basic.\n",arg);

        return type;
    }

    CcStatsType_t get_cc_stats_type(const char*arg) {
        CcStatsType_t type = CC_STATS_NONE;
        bool all_ok = false;

        check_arg_and_set(type,all_ok,arg,"none"       ,CC_STATS_NONE);
        check_arg_and_set(type,all_ok,arg,"probability",CC_STATS_PROB);
        check_arg_and_set(type,all_ok,arg,"sigma"      ,CC_STATS_SIGMA);
        check_arg_and_set(type,all_ok,arg,"wgt_avg"    ,CC_STATS_WGT_AVG);

        if( !all_ok )
            fprintf(stderr,"Invalid cc statistics type %s. Options are: none, probability or sigma. Defaulting to none.\n",arg);

        return type;
    }

    NormalizationType_t get_norm_type(const char*arg) {
        NormalizationType_t type = NO_NORM;
        bool all_ok = false;

        check_arg_and_set(type,all_ok,arg,"none"                 ,NO_NORM        );
        check_arg_and_set(type,all_ok,arg,"zero_mean"            ,ZERO_MEAN      );
        check_arg_and_set(type,all_ok,arg,"zero_mean_proj_weight",ZERO_MEAN_W_STD); /// ToBeDeprecated
        check_arg_and_set(type,all_ok,arg,"zero_mean_one_std"    ,ZERO_MEAN_1_STD);
        check_arg_and_set(type,all_ok,arg,"poisson_raw"          ,GAT_RAW);
        check_arg_and_set(type,all_ok,arg,"poisson_normal"       ,GAT_NORMAL);

        if( !all_ok )
            fprintf(stderr,"Invalid normalization type %s. Options are: none, zero_mean, zero_mean_proj_weight and zero_mean_one_std. Defaulting to none.\n",arg);

        return type;
    }

    WeightingType_t get_weighting_type(const char*arg) {
        WeightingType_t type = WGT_NONE;
        bool all_ok = false;

        check_arg_and_set(type,all_ok,arg,"none"      ,WGT_NONE);
        check_arg_and_set(type,all_ok,arg,"particle"  ,WGT_3D  );
        check_arg_and_set(type,all_ok,arg,"projection",WGT_2D  );
        check_arg_and_set(type,all_ok,arg,"3DCC"      ,WGT_3DCC);
        check_arg_and_set(type,all_ok,arg,"2DCC"      ,WGT_2DCC);

        if( !all_ok )
            fprintf(stderr,"Invalid weighting type %s. Options are: none, particle, projection, 3DCC and 2DCC. Defaulting to none.\n",arg);

        return type;
    }

    CtfAlignmentType_t get_ali_ctf_type(const char*arg) {
        CtfAlignmentType_t type = ALI_CTF_DISABLED;
        bool all_ok = false;

        check_arg_and_set(type,all_ok,arg,"none"        ,ALI_CTF_DISABLED        );
        check_arg_and_set(type,all_ok,arg,"phase_flip"  ,ALI_CTF_PHASE_FLIP      );
        check_arg_and_set(type,all_ok,arg,"on_reference",ALI_CTF_ON_REFERENCE    );
        check_arg_and_set(type,all_ok,arg,"on_substack" ,ALI_CTF_ON_SUBSTACK     );
        check_arg_and_set(type,all_ok,arg,"wiener_ssnr" ,ALI_CTF_ON_SUBSTACK_SSNR);
        check_arg_and_set(type,all_ok,arg,"cfsc"        ,ALI_CTF_ON_REFERENCE    ); /// ToBeDeprecated

        if( !all_ok )
            fprintf(stderr,"Invalid ctf correction type %s. Options are: none, phase_flip, on_reference, on_substack or wiener_ssnr. Defaulting to on_reference.\n",arg);

        return type;
    }

    CtfInversionType_t get_inv_ctf_type(const char*arg) {
        CtfInversionType_t type = INV_NO_INV;
        bool all_ok = false;

        check_arg_and_set(type,all_ok,arg,"none"       ,INV_NO_INV     );
        check_arg_and_set(type,all_ok,arg,"phase_flip" ,INV_PHASE_FLIP );
        check_arg_and_set(type,all_ok,arg,"wiener"     ,INV_WIENER     );
        check_arg_and_set(type,all_ok,arg,"wiener_ssnr",INV_WIENER_SSNR);
        check_arg_and_set(type,all_ok,arg,"pre_wiener" ,INV_PRE_WIENER );

        if( !all_ok )
            fprintf(stderr,"Invalid ctf correction type %s. Options are: none, phase_flip, wiener, pre_wiener and wiener_ssnr. Defaulting to wiener.\n",arg);

        return type;
    }

    CtfProjectionType_t get_proj_ctf_type(const char*arg) {
        CtfProjectionType_t type = PRJ_STANDARD;
        bool all_ok = false;

        check_arg_and_set(type,all_ok,arg,"none"    ,PRJ_NO_CTF  );
        check_arg_and_set(type,all_ok,arg,"standard",PRJ_STANDARD);
        check_arg_and_set(type,all_ok,arg,"basic"   ,PRJ_STANDARD);

        if( !all_ok )
            fprintf(stderr,"Invalid ctf type %s. Options are: none, standard and basic. Defaulting to basic.\n",arg);

        return type;
    }

    OffsetType_t get_offset_type(const char*arg) {
        OffsetType_t type = ELLIPSOID;
        bool all_ok = false;

        check_arg_and_set(type,all_ok,arg,"ellipsoid",ELLIPSOID);
        check_arg_and_set(type,all_ok,arg,"cylinder" ,CYLINDER );
        check_arg_and_set(type,all_ok,arg,"circle"   ,CIRCLE   );
        check_arg_and_set(type,all_ok,arg,"cuboid"   ,CUBOID   );
        check_arg_and_set(type,all_ok,arg,"rectangle",RECTANGLE);

        if( !all_ok )
            fprintf(stderr,"Invalid offset type %s. Options are: ellipsoid, cylinder or cuboid for 3D search, or circle and rectangle for 2D search. Defaulting to ellipsoid.\n",arg);

        return type;
    }

    OffsetSpace_t get_offset_space(const char*arg) {
        OffsetSpace_t type = REFERENCE_SPACE;
        bool all_ok = false;

        check_arg_and_set(type,all_ok,arg,"reference",REFERENCE_SPACE);
        check_arg_and_set(type,all_ok,arg,"tomogram" ,TOMOGRAM_SPACE );

        if( !all_ok )
            fprintf(stderr,"Invalid offset space %s. Options are: reference or tomogram. Defaulting to reference.\n",arg);

        return type;
    }

    int get_even_number(const char*arg) {
        int rslt = atoi(arg);
        return rslt + (0x01&rslt); /// Force to be even number.
    }

    bool get_bool(const char*arg) {
        int rslt = atoi(arg);
        return (rslt>0);
    }

    void get_single_pair(float&val_a,float&val_b,const char*arg) {
        int len_arg = strlen(arg);
        char buffer[len_arg+1];
        for(int i=0;i<len_arg;i++){
            buffer[i] = arg[i];
            if(arg[i]==',')
                buffer[i] = ' ';
        }
        buffer[len_arg] = 0;
        int validate = sscanf(buffer,"%f %f",&val_a,&val_b);
        if( validate != 2 ) {
            fprintf(stderr,"Error parsing single pair: %s\n",arg);
            exit(1);
        }
    }

    void get_single_trio(float&val_a,float&val_b,float&val_c,const char*arg) {
            int len_arg = strlen(arg);
            char buffer[len_arg+1];
            for(int i=0;i<len_arg;i++){
                buffer[i] = arg[i];
                if(arg[i]==',')
                    buffer[i] = ' ';
            }
            buffer[len_arg] = 0;
            int validate = sscanf(buffer,"%f %f %f",&val_a,&val_b,&val_c);
            if( validate != 3 ) {
                fprintf(stderr,"Error parsing single trio: %s\n",arg);
                exit(1);
            }
        }
    
    void get_single_quad(float&val_a,float&val_b,float&val_c,float&val_d,const char*arg) {
        int len_arg = strlen(arg);
        char buffer[len_arg+1];
        for(int i=0;i<len_arg;i++){
            buffer[i] = arg[i];
            if(arg[i]==',')
                buffer[i] = ' ';
        }
        buffer[len_arg] = 0;
        int validate = sscanf(buffer,"%f %f %f %f",&val_a,&val_b,&val_c,&val_d);
        if( validate != 4 ) {
            fprintf(stderr,"Error parsing single pair: %s\n",arg);
            exit(1);
        }
    }

    void get_uint32_pair(uint32&val_a,uint32&val_b,const char*arg) {
        int len_arg = strlen(arg);
        char buffer[len_arg+1];
        for(int i=0;i<len_arg;i++){
            buffer[i] = arg[i];
            if(arg[i]==',')
                buffer[i] = ' ';
        }
        buffer[len_arg] = 0;
        int validate = sscanf(buffer,"%u %u",&val_a,&val_b);
        if( validate != 2 ) {
            fprintf(stderr,"Error parsing uint32 pair: %s\n",arg);
            exit(1);
        }
    }

    int get_list_integers(unsigned int p_int[],const char*arg) {
        int len_arg = strlen(arg);
        char buffer[len_arg+1];
        for(int i=0;i<len_arg;i++){
            buffer[i] = arg[i];
            if(arg[i]==',')
                buffer[i] = ' ';
        }
        int rslt = 0;
        int offset = 0;
        char*p_buffer = (char*)buffer;
        while (sscanf(p_buffer," %u%n", p_int+rslt, &offset) == 1) {
            p_buffer += offset;
            rslt++;
        }
        return rslt;
    }
}

