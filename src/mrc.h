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

#ifndef MRC_H
#define MRC_H

#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <string>
#include "datatypes.h"
#include "math_cpu.h"

using namespace std;

namespace Mrc {

    float get_apix (const char*mapname) {
        FILE*fp=fopen(mapname,"rb");
        if( fp == NULL ) {
            fprintf(stderr,"Error opening file %s.\n",mapname);
            exit(1);
        }
        fseek(fp,28,SEEK_SET);
        uint32 mx;
        fread((void*)(&mx),sizeof(uint32_t),1,fp);
        fseek(fp,40,SEEK_SET);
        float xlen;
        fread((void*)(&xlen),sizeof(float),1,fp);
        fclose(fp);
        if( xlen == 0 )
            return 1.0f;
        else
            return xlen/mx;
    }

    void set_apix (const char*mapname, const float apix, const uint32 X, const uint32 Y, const uint32 Z) {
        FILE*fp=fopen(mapname,"r+");
        if( fp == NULL ) {
            fprintf(stderr,"Error opening file %s.\n",mapname);
            exit(1);
        }
        uint32_t grid[3];
        grid[0] = X;
        grid[1] = Y;
        grid[2] = Z;
        fseek(fp,28,SEEK_SET);
        fwrite(grid,sizeof(uint32_t),3,fp);
        float apix_arr[3];
        apix_arr[0] = apix*grid[0];
        apix_arr[1] = apix*grid[1];
        apix_arr[2] = apix*grid[2];
        fseek(fp,40,SEEK_SET);
        fwrite(apix_arr,sizeof(float),3,fp);
        fclose(fp);
    }

    void set_ispg (const char*mapname, const uint32 value) {
        FILE*fp=fopen(mapname,"r+");
        if( fp == NULL ) {
            fprintf(stderr,"Error opening file %s.\n",mapname);
            exit(1);
        }
        fseek(fp,88,SEEK_SET);
        fwrite(&value,sizeof(uint32_t),1,fp);
        fclose(fp);
    }

    void set_as_stack(const char*mapname) {
        set_ispg(mapname,0);
    }

    void set_as_volume(const char*mapname) {
        set_ispg(mapname,1);
    }

    bool is_mode_float(const char*mapname) {
        FILE*fp=fopen(mapname,"rb");
        if( fp == NULL ) {
            fprintf(stderr,"Error opening file %s.\n",mapname);
            exit(1);
        }
        fseek(fp,12,SEEK_SET);
        uint32 mode;
        fread((void*)(&mode),sizeof(uint32_t),1,fp);
        fclose(fp);
        if( mode == 2 )
            return true;
        else
            return false;
    }

    void read_size(uint32&X, uint32&Y, uint32&Z, const char*mapname) {
        FILE*fp=fopen(mapname,"rb");
        if( fp == NULL ) {
            fprintf(stderr,"Error opening file %s.\n",mapname);
            exit(1);
        }
        fseek(fp,0,SEEK_SET);
        uint32 buf[3];
        fread((void*)buf,sizeof(uint32_t),3,fp);
        fclose(fp);
        X = buf[0];
        Y = buf[1];
        Z = buf[2];
    }

    void read(float*buffer, const uint32 X, const uint32 Y, const uint32 Z, const char*mapname) {
        if(!is_mode_float(mapname)) {
            fprintf(stderr,"Error: File %s is not mode FLOAT32.\n",mapname);
            exit(1);
        }
        FILE*fp=fopen(mapname,"rb");
        if( fp == NULL ) {
            fprintf(stderr,"Error opening file %s.\n",mapname);
            exit(1);
        }
        uint32 offset = 0;
        fseek(fp,92,SEEK_SET);
        fread((void*)(&offset),sizeof(uint32),1,fp);
        fseek(fp,1024+offset,SEEK_SET);
        size_t num_el = X*Y*Z;
        size_t read_el = fread((void*)buffer,sizeof(single),num_el,fp);
        fclose(fp);
        if( num_el != read_el ) {
            fprintf(stderr,"Error: File %s truncated.\n",mapname);
            exit(1);
        }
    }

    float *read(uint32&X, uint32&Y, uint32&Z, const char*mapname) {
        read_size(X,Y,Z,mapname);
        single *rslt = new single[X*Y*Z];
        read(rslt,X,Y,Z,mapname);
        return rslt;
    }
    
    void write(const single *data, const uint32 X, const uint32 Y, const uint32 Z, const char*mapname,bool fill_statistics=true) {
        FILE*fp=fopen(mapname,"wb");
        if( fp == NULL ) {
            fprintf(stderr,"Error creating file %s.\n",mapname);
            exit(1);
        }
        uint32 header[256];
        for(int i=0;i<256;i++) header[i] = 0;
        header[0]  = X;
        header[1]  = Y;
        header[2]  = Z;
        header[3]  = 2;
        header[7]  = 1;
        header[8]  = 1;
        header[9]  = 1;
        header[13] = 0x42b40000; // 90.0 in hexadecimal notation.
        header[14] = 0x42b40000; // 90.0 in hexadecimal notation.
        header[15] = 0x42b40000; // 90.0 in hexadecimal notation.
        header[16] = 1;
        header[17] = 2;
        header[18] = 3;
        header[27] = 20140;
        header[52] = 0x2050414D;
        header[53] = 0x00004444;
        if(fill_statistics) {
            float stats[4];
            Math::get_min_max_avg_std(stats[0],stats[1],stats[2],stats[3],data,X*Y*Z);
            uint32*tmp = (uint32*)stats;
            header[19] = tmp[0];
            header[20] = tmp[1];
            header[21] = tmp[2];
            header[54] = tmp[3];
        }
        fwrite((void*)header,sizeof(uint32),256,fp);
        fwrite((void*)data,sizeof(single),X*Y*Z,fp);
        fclose(fp);
    }

    class SequentialWriter {

    public:
        int N;
        int K;
        float vmax;
        float vmin;
        float vavg;
        float vstd;
        float apix;
        FILE *fp;


        SequentialWriter(const char*filename,int in_N,float in_apix) {
            fp = fopen(filename,"wb");
            if( fp == NULL ) {
                fprintf(stderr,"Error creating file %s.\n",filename);
                exit(1);
            }
            N  = in_N;
            K  = 0;
            vmax = -9999.9;
            vmin =  9999.9;
            vavg = 0.0;
            vstd = 0.0;
            apix = in_apix;

            /// init clean header
            uint32 header[256];
            for(int i=0;i<256;i++) header[i] = 0;
            fwrite((void*)header,sizeof(uint32),256,fp);
        }

        ~SequentialWriter() {
            float stats[4];
            stats[0] = vmin;
            stats[1] = vmax;
            stats[2] = vavg;
            stats[3] = vstd;
            uint32*tmp = (uint32*)stats;

            uint32 header[256];
            for(int i=0;i<256;i++) header[i] = 0;
            header[0]  = N;
            header[1]  = N;
            header[2]  = K;
            header[3]  = 2;
            header[7]  = N;
            header[8]  = N;
            header[9]  = K;
            float cell[3] = { apix*(float)N, apix*(float)N, 1.0f };
            uint32*cell_bits = (uint32*)cell;
            header[10] = cell_bits[0];
            header[11] = cell_bits[1];
            header[12] = cell_bits[2];
            header[13] = 0x42b40000; // 90.0 in hexadecimal notation.
            header[14] = 0x42b40000; // 90.0 in hexadecimal notation.
            header[15] = 0x42b40000; // 90.0 in hexadecimal notation.
            header[16] = 1;
            header[17] = 2;
            header[18] = 3;
            header[27] = 20140;
            header[52] = 0x2050414D;
            header[53] = 0x00004444;
            header[19] = tmp[0];
            header[20] = tmp[1];
            header[21] = tmp[2];
            header[54] = tmp[3];

            fseek(fp,0,SEEK_SET);
            fwrite((void*)header,sizeof(uint32),256,fp);
            fclose(fp);
        }

        void push_frame(const float*in_data) {
            fwrite((void*)in_data,sizeof(single),N*N,fp);

            /// update using parallel algorithm for track stats
            float n_a = K*N*N;
            float x_a = vavg;
            float M_a = fabs((vstd*vstd)*(n_a-1));

            float cmin,cmax,cavg,cstd;
            Math::get_min_max_avg_std(cmin,cmax,cavg,cstd,in_data,N*N);
            float n_b = N*N;
            float x_b = cavg;
            float M_b = fabs((cstd*cstd)*(n_b-1));

            float n_ab  = n_a + n_b;
            float delta = x_b - x_a;
            float M_ab  = M_a + M_b + (delta*delta)*n_a*n_b/n_ab;

            vavg = (n_a*x_a + n_b*x_b)/n_ab;
            vstd = (n_ab > 1) ? sqrtf( M_ab/(n_ab-1) ) : 0.0f;

            vmax = fmax(cmax,vmax);
            vmin = fmin(cmin,vmin);

            K++;

        }

    };


}

#endif 

