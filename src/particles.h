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

#ifndef PARTICLES_H
#define PARTICLES_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "datatypes.h"
#include "io.h"
#include "math_cpu.h"
#include "memory.h"

class Particle {

public:
    PtclInf *info;
    Vec3    *ali_eu;  /// radians
    Vec3    *ali_t;   /// Angstroms
    single  *ali_cc;
    single  *ali_w;
    Vec3    *prj_eu;  /// radians
    Vec2    *prj_t;   /// Angstroms
    single  *prj_cc;
    single  *prj_w;
    Defocus *def;

    uint32  n_proj;
    uint32  n_refs;

public:
    Particle() {
        info   = NULL;
        ali_eu = NULL;
        ali_t  = NULL;
        ali_cc = NULL;
        ali_w  = NULL;
        prj_eu = NULL;
        prj_t  = NULL;
        prj_cc = NULL;
        prj_w  = NULL;
        def    = NULL;
    }

    void set(uint8*ptr,uint32 proj,uint32 refs) {

        n_proj = proj;
        n_refs = refs;

        info   = (PtclInf*)ptr;

        ali_eu = (Vec3*  )(info+1);
        ali_t  =           ali_eu + n_refs;
        ali_cc = (single*)(ali_t  + n_refs);
        ali_w  = (single*)(ali_cc + n_refs);

        prj_eu = (Vec3*  )(ali_w  + n_refs);
        prj_t  = (Vec2*  )(prj_eu + n_proj);
        prj_cc = (single*)(prj_t  + n_proj);
        prj_w  =           prj_cc + n_proj;

        def    = (Defocus*)(prj_w + n_proj);
    }

    uint32& ptcl_id() {
        return info->ptcl_id;
    }

    uint32& tomo_id() {
        return info->tomo_id;
    }

    uint32& tomo_cix() {
        return info->tomo_cix;
    }

    Vec3&   pos() {
        return info->pos;
    }

    uint32& ref_cix() {
        return info->ref_cix;
    }

    uint32& half_id() {
        return info->half_id;
    }

    single& extra_1() {
        return info->extra_1;
    }

    single& extra_2() {
        return info->extra_2;
    }

    void print() {
        printf("  ID:         %d\n" ,ptcl_id());
        printf("  Tomo ID:    %d (%d)\n",tomo_id(),tomo_cix());
        printf("  Position:   %.1f,%.1f,%.1f\n",pos().x,pos().y,pos().z);
        printf("  Ref. idx:   %d\n",ref_cix());
        printf("  Half ID:    %d\n",half_id());
        printf("  Extras:     %f %f\n",extra_1(),extra_2());
        printf("  References:\n");
        for( uint32 refs = 0; refs < n_refs; refs++ ) {
            printf("    %2d [",refs);
            printf( "%8.3f", ali_eu[refs].x*180/M_PI);
            printf(" %8.3f", ali_eu[refs].y*180/M_PI);
            printf(" %8.3f", ali_eu[refs].z*180/M_PI);
            printf("] [");
            printf( "%7.2f", ali_t[refs].x);
            printf(" %7.2f", ali_t[refs].y);
            printf(" %7.2f", ali_t[refs].z);
            printf("] (%f %f)\n",ali_cc[refs],ali_w[refs]);
        }

        printf("  2D Refine:\n");
        for( uint32 proj = 0; proj < n_proj; proj++ ) {
            printf("    %2d [",proj);
            printf( "%8.3f", prj_eu[proj].x*180/M_PI);
            printf(" %8.3f", prj_eu[proj].y*180/M_PI);
            printf(" %8.3f", prj_eu[proj].z*180/M_PI);
            printf("] [");
            printf( "%7.2f", prj_t[proj].x);
            printf(" %7.2f", prj_t[proj].y);
            printf("] (%f)",prj_cc[proj]);
            printf("(%f)\n",prj_w [proj]);
        }

        printf("  Defocus:\n");
        for( uint32 proj = 0; proj < n_proj; proj++ ) {
            printf("    %2d |",proj);
            printf(" %10.2f %10.2f %8.3f |",def[proj].U,def[proj].V,def[proj].angle);
            printf(" %7.2f %7.2f %7.2f |"  ,def[proj].Bfactor,def[proj].ExpFilt,def[proj].max_res);
            printf(" %8.4f\n",def[proj].score);
        }
    }

public:
    void copy(Particle&ptcl_in) {
        memcpy(info  ,ptcl_in.info  ,sizeof(PtclInf)       );
        memcpy(ali_eu,ptcl_in.ali_eu,sizeof(Vec3   )*n_refs);
        memcpy(ali_t ,ptcl_in.ali_t ,sizeof(Vec3   )*n_refs);
        memcpy(ali_cc,ptcl_in.ali_cc,sizeof(single )*n_refs);
        memcpy(ali_w ,ptcl_in.ali_w ,sizeof(single )*n_refs);
        memcpy(prj_eu,ptcl_in.prj_eu,sizeof(Vec3   )*n_proj);
        memcpy(prj_t ,ptcl_in.prj_t ,sizeof(Vec2   )*n_proj);
        memcpy(prj_cc,ptcl_in.prj_cc,sizeof(single )*n_proj);
        memcpy(prj_w ,ptcl_in.prj_w ,sizeof(single )*n_proj);
        memcpy(def   ,ptcl_in.def   ,sizeof(Defocus)*n_proj);
    }

};

class Particles {

public:
    uint32 n_ptcl;
    uint32 n_proj;
    uint32 n_refs;
    uint8  *p_raw;
    uint32 n_bytes;
    
public:
    Particles() {
        n_ptcl = 0;
        n_proj = 0;
        n_refs = 0;
        n_bytes = 0;
    }

    virtual ~Particles() = default;

    bool get(Particle&ptcl,uint32 ix=0) {
        uint64 offset = n_bytes;
        offset *= ix;
        if( ix < n_ptcl ) {
            ptcl.set(p_raw+offset,n_proj,n_refs);
            return true;
        }
        return false;
    }

public:
    static bool check_signature(const char*filename) {

        FILE*fp = fopen(filename,"rb");

        if( fp == NULL ) {
            fprintf(stderr,"Reading %s: cannot open file.\n",filename);
            return false;
        }

        char signature[9] = {0};
        if( !IO::check_fread(signature, sizeof(char), 8, fp) ) {
            fprintf(stderr,"Reading %s: truncated file while reading signature.\n",filename);
            fclose(fp);
            return false;
        }

        if( strcmp(signature,"SsaPtcl1") != 0) {
            fprintf(stderr,"Trying to read %s: wrong file signature %s.\n",filename,signature);
            fclose(fp);
            return false;
        }

        fclose(fp);
        return true;
    }

protected:
    void allocate(uint32 ptcl,uint32 proj,uint32 refs) {
        n_bytes = sizeof(PtclInf)
                + refs*(sizeof(Vec3)+sizeof(Vec3)+sizeof(float)+sizeof(float)) /// 3D ALIGNMENT PER CLASS/REF
                + proj*(sizeof(Vec3)+sizeof(Vec2)+sizeof(float)+sizeof(float)) /// 2D ALIGNMENT PER PROJECTION
                + proj*(sizeof(Defocus));                                      /// DEFOCUS PER PROJECTION

        size_t total_bytes = n_bytes;
        total_bytes *= ptcl;

        p_raw = (uint8*)malloc(total_bytes);
        if( p_raw == NULL ) {
            fprintf(stderr,"Error allocating memory for the particles descriptors (%ld)\n",n_bytes*ptcl);
            exit(1);
        }
        memset(p_raw,0,total_bytes);
    }

};

class ParticlesRW : public Particles {

public:
    ParticlesRW(const char*filename) {

        FILE*fp = fopen(filename,"rb");
        if( fp == NULL ) {
            fprintf(stderr,"Reading %s: cannot open file.\n",filename);
            exit(1);
        }

        char signature[9] = {0};
        if( !IO::check_fread(signature, sizeof(char), 8, fp) ) {
            fprintf(stderr,"Reading %s: truncated file while reading signature.\n",filename);
            exit(1);
        }

        if( strcmp(signature,"SsaPtcl1") != 0) {
            fprintf(stderr,"Trying to read %s: wrong file signature %s.\n",filename,signature);
            exit(1);
        }

        uint32_t lengths[3] = {0};
        if( !IO::check_fread(lengths, sizeof(uint32_t), 3, fp) ) {
            fprintf(stderr,"Reading %s: truncated file while reading sizes.\n",filename);
            exit(1);
        }
        n_ptcl = lengths[0];
        n_proj = lengths[1];
        n_refs = lengths[2];

        allocate(n_ptcl,n_proj,n_refs);

        if( !IO::check_fread(p_raw,n_bytes,n_ptcl,fp) ) {
            fprintf(stderr,"Reading %s: truncated file while reading particles information.\n",filename);
            exit(1);
        }

        fclose(fp);
    }

    ~ParticlesRW() {
            free(p_raw);
    }

    void save(const char*filename) {
        FILE*fp = fopen(filename,"wb");
        if( fp == NULL ) {
            fprintf(stderr,"Cannot open file %s for writing.\n",filename);
            return;
        }
        char signature[] = "SsaPtcl1";
        uint32_t lengths[3];
        lengths[0] = n_ptcl;
        lengths[1] = n_proj;
        lengths[2] = n_refs;
        fwrite(signature, sizeof(char), 8, fp);
        fwrite(lengths, sizeof(uint32), 3, fp);        
        fwrite(p_raw,n_bytes,n_ptcl,fp);
        fclose(fp);
    }

};

class ParticlesInStream : public Particles {

protected:
    FILE   *fp;
    uint32 counter;

public:
    ParticlesInStream(const char*filename) {

        fp = fopen(filename,"rb");
        if( fp == NULL ) {
            fprintf(stderr,"Reading %s: cannot open file.\n",filename);
            exit(1);
        }

        char signature[9] = {0};
        if( !IO::check_fread(signature, sizeof(char), 8, fp) ) {
            fprintf(stderr,"Reading %s: truncated file while reading signature.\n",filename);
            exit(1);
        }

        if( strcmp(signature,"SsaPtcl1") != 0) {
            fprintf(stderr,"Trying to read %s: wrong file signature %s.\n",filename,signature);
            exit(1);
        }

        uint32_t lengths[3] = {0};
        if( !IO::check_fread(lengths, sizeof(uint32_t), 3, fp) ) {
            fprintf(stderr,"Reading %s: truncated file while reading sizes.\n",filename);
            exit(1);
        }
        n_ptcl = lengths[0];
        n_proj = lengths[1];
        n_refs = lengths[2];

        allocate(1,n_proj,n_refs);

        counter = 0;
    }

    ~ParticlesInStream() {
        fclose(fp);
        free(p_raw);
    }

    bool get(Particle&ptcl,uint32 ix=0) {
        if( ix < 1 ) {
            ptcl.set(p_raw,n_proj,n_refs);
            return true;
        }
        return false;
    }

    bool read_buffer() {
        if( counter >= n_ptcl )
            return false;
        if( IO::check_fread(p_raw,n_bytes,1,fp) ) {
            counter++;
            return true;
        }
        fprintf(stderr,"Reading particles: truncated file while reading particles information.\n");
        return false;
    }

};

class ParticlesOutStream : public Particles {

protected:
    FILE   *fp;
    uint32 counter;

public:
    ParticlesOutStream(const char*filename,uint32 proj,uint32 refs) {
        n_ptcl = 1;
        n_proj = proj;
        n_refs = refs;

        fp = fopen(filename,"wb");
        if( fp == NULL ) {
            fprintf(stderr,"Writing %s: cannot create file.\n",filename);
            exit(1);
        }
        
        char signature[] = "SsaPtcl1";
        uint32_t lengths[3];
        lengths[0] = 0;
        lengths[1] = n_proj;
        lengths[2] = n_refs;
        fwrite(signature, sizeof(char), 8, fp);
        fwrite(lengths, sizeof(uint32), 3, fp);

        allocate(1,n_proj,n_refs);

        counter = 0;
    }

    ~ParticlesOutStream() {
        fseek(fp,8,SEEK_SET);
        fwrite(&counter,sizeof(uint32),1,fp);
        fclose(fp);
        free(p_raw);
    }

    void write_buffer() {
        fwrite(p_raw,n_bytes,1,fp);
        counter++;
    }

};

class ParticlesMem : public Particles {
public:
    ParticlesMem(uint32 ptcl,uint32 proj,uint32 refs) {
        n_ptcl = ptcl;
        n_proj = proj;
        n_refs = refs;
        allocate(ptcl,proj,refs);
    }

    ~ParticlesMem() {
        free(p_raw);
    }
};

class ParticlesSubset : public Particles {
public:
    void set(Particles&ptcls_in,uint32 offset,uint32 length) {
        n_proj  = ptcls_in.n_proj;
        n_refs  = ptcls_in.n_refs;
        n_bytes = ptcls_in.n_bytes;

        uint64 w_offset = n_bytes;
        w_offset *= offset;
        p_raw   = ptcls_in.p_raw + w_offset;

        if( offset < ptcls_in.n_ptcl ) {
            uint32 w_length = offset + length;
            if( w_length > ptcls_in.n_ptcl )
                w_length = ptcls_in.n_ptcl;
            w_length = w_length - offset;
            n_ptcl   = w_length;
        }
        else {
            p_raw  = ptcls_in.p_raw;
            n_ptcl = 0;
        }
    }
};

#endif /// PARTICLES_H



