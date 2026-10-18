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

#ifndef DENSE_REPORTER_H
#define DENSE_REPORTER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cctype>
#include <vector>
#include <unistd.h>

#include "datatypes.h"
#include "gpu.h"
#include "mrc.h"
#include "math_cpu.h"
#include "particles.h"
#include "tomogram.h"
#include "angles_provider.h"
#include "points_provider.h"
#include "aligner_args.h"

namespace DenseKernels {

__global__ void clear(float*best,int*idx,double*mean,double*m2,const int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if( i < n ) {
        best[i] = -INFINITY;
        idx[i]  = -1;
        mean[i] = 0;
        m2[i]   = 0;
    }
}

__global__ void update(float*best,int*idx,double*mean,double*m2,const float*cc,const int ang,const double count,const int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if( i < n ) {
        float v = cc[i];
        if( v > best[i] ) {
            best[i] = v;
            idx[i]  = ang;
        }
        double d = v - mean[i];
        mean[i] += d/count;
        m2[i]   += d*(v - mean[i]);
    }
}

/// Per voxel: z = (best - mean)/sd of each of its points; 'max' keeps the point with the highest
/// z, 'mean' averages cc and z (angle of the best point); NaN without points.
__global__ void finalize(float*vol,const float*best,const int*idx,const double*mean,const double*m2,
                         const int*vox_start,const int*vox_pts,const double count,const int n_vox,const int reduce) {
    int v = blockIdx.x*blockDim.x + threadIdx.x;
    if( v < n_vox ) {
        int s = vox_start[v], e = vox_start[v+1];
        if( s == e ) {
            vol[v] = vol[n_vox+v] = vol[2*n_vox+v] = nanf("");
            return;
        }
        double z_best = -INFINITY, cc_acc = 0, z_acc = 0;
        int p_best = vox_pts[s];
        for(int j=s;j<e;j++) {
            int p = vox_pts[j];
            double sd = sqrt(m2[p]/count);
            double z  = ( sd > SUSAN_FLOAT_TOL ) ? (best[p]-mean[p])/sd : 0.0;
            if( z > z_best ) { z_best = z; p_best = p; }
            cc_acc += best[p];
            z_acc  += z;
        }
        if( reduce == DENSE_REDUCE_MEAN ) {
            vol[v]       = (float)(cc_acc/(e-s));
            vol[n_vox+v] = (float)(z_acc/(e-s));
        }
        else {
            vol[v]       = best[p_best];
            vol[n_vox+v] = (float)z_best;
        }
        vol[2*n_vox+v] = (float)idx[p_best];
    }
}

}

namespace DenseIO {

/// Voxel grid of the offset points (rounded half away from zero, as the block centre): lower
/// corner and shape (x, y, z).
inline void grid(int3&lo,int3&shape,const Vec3*pts,const int n_pts) {
    int3 hi;
    for(int i=0;i<n_pts;i++) {
        int x = (int)roundf(pts[i].x), y = (int)roundf(pts[i].y), z = (int)roundf(pts[i].z);
        if( i == 0 ) { lo = make_int3(x,y,z); hi = lo; }
        lo.x = std::min(lo.x,x); lo.y = std::min(lo.y,y); lo.z = std::min(lo.z,z);
        hi.x = std::max(hi.x,x); hi.y = std::max(hi.y,y); hi.z = std::max(hi.z,z);
    }
    shape = make_int3(hi.x-lo.x+1,hi.y-lo.y+1,hi.z-lo.z+1);
}

inline void block_file(char*out,const char*prefix,const uint32 ptcl_id,const char*name) {
    snprintf(out,SUSAN_FILENAME_LENGTH,"%s_b%07d_%s.mrc",prefix,ptcl_id,name);
}

/// MRC with the block corner in nxstart/nystart/nzstart and the origin; statistics skip NaN.
inline void write_block(const float*data,const int3 shape,const char*filename,const float apix,const int3 corner) {
    FILE*fp = fopen(filename,"wb");
    if( fp == NULL ) {
        fprintf(stderr,"Error creating file %s.\n",filename);
        exit(1);
    }
    size_t n = (size_t)shape.x*shape.y*shape.z;
    double s=0,s2=0;
    float vmin=INFINITY,vmax=-INFINITY;
    size_t cnt=0;
    for(size_t i=0;i<n;i++) {
        if( std::isnan(data[i]) ) continue;
        vmin = fminf(vmin,data[i]); vmax = fmaxf(vmax,data[i]);
        s += data[i]; s2 += (double)data[i]*data[i]; cnt++;
    }
    float avg = cnt ? (float)(s/cnt) : 0, rms = cnt ? (float)sqrt(fmax(s2/cnt-(s/cnt)*(s/cnt),0.0)) : 0;
    if( !cnt ) { vmin = vmax = 0; }

    uint32 header[256];
    memset(header,0,sizeof(header));
    int32_t*ih = (int32_t*)header;
    float  *fh = (float*)header;
    ih[0] = shape.x; ih[1] = shape.y; ih[2] = shape.z;
    ih[3] = 2;
    ih[4] = corner.x; ih[5] = corner.y; ih[6] = corner.z;
    ih[7] = shape.x; ih[8] = shape.y; ih[9] = shape.z;
    fh[10] = apix*shape.x; fh[11] = apix*shape.y; fh[12] = apix*shape.z;
    fh[13] = fh[14] = fh[15] = 90.0f;
    ih[16] = 1; ih[17] = 2; ih[18] = 3;
    fh[19] = vmin; fh[20] = vmax; fh[21] = avg;
    ih[27] = 20140;
    fh[49] = apix*corner.x; fh[50] = apix*corner.y; fh[51] = apix*corner.z;
    header[52] = 0x2050414D;
    header[53] = 0x00004444;
    fh[54] = rms;
    fwrite(header,sizeof(uint32),256,fp);
    fwrite(data,sizeof(float),n,fp);
    fclose(fp);
}

/// {prefix}.json: run parameters, the angle table (level-0 search rotations, SUSAN ZYZ, radians)
/// and the blocks. Absolute pose of a voxel: R(ali_eu of its block) R(table[ang]).
inline void write_manifest(const ArgsAli::Info&info,Particles&ptcls,Tomograms&tomos) {
    char filename[SUSAN_FILENAME_LENGTH];
    snprintf(filename,SUSAN_FILENAME_LENGTH,"%s.json",info.output);
    FILE*fp = fopen(filename,"w");
    if( fp == NULL ) {
        fprintf(stderr,"Error creating file %s.\n",filename);
        exit(1);
    }

    uint32 n_pts;
    Vec3*pts = PointsProvider::create(n_pts,info.off_type,info.off_x,info.off_y,info.off_z,info.off_s);
    int3 lo,shape;
    grid(lo,shape,pts,n_pts);
    delete [] pts;

    fprintf(fp,"{\n \"format\": \"susan-dense\",\n \"version\": 1,\n");
    fprintf(fp," \"params\": {\"box_size\": %d, \"pad_size\": %d, \"bandpass_fpix\": [%.6g, %.6g], \"rolloff\": %.6g, ",
            info.box_size,info.pad_size,info.fpix_min,info.fpix_max,info.fpix_roll);
    fprintf(fp,"\"cone\": [%.6g, %.6g], \"inplane\": [%.6g, %.6g], \"pseudo_symmetry\": \"%s\", ",
            info.cone_range,info.cone_step,info.inplane_range,info.inplane_step,info.pseudo_sym);
    fprintf(fp,"\"off_type\": %d, \"off_params\": [%.6g, %.6g, %.6g, %.6g], \"dilate\": %.6g, \"dilate_radius\": %d, ",
            info.off_type,info.off_x,info.off_y,info.off_z,info.off_s,info.dilate,info.dilate_radius);
    fprintf(fp,"\"cc_type\": %d, \"ctf_type\": %d, \"norm_type\": %d, \"pad_type\": %d, \"reduce\": \"%s\", ",
            info.cc_type,info.ctf_type,info.norm_type,info.pad_type,(info.dense_reduce==DENSE_REDUCE_MEAN)?"mean":"max");
    fprintf(fp,"\"refs_file\": \"%s\", \"tomos_file\": \"%s\", \"ptcls_in\": \"%s\", \"pose\": \"R(ali_eu) @ R(table[ang])\"},\n",
            info.refs_file,info.tomo_file,info.ptcls_in);

    AnglesProvider ang_prov;
    ang_prov.cone_range    = info.cone_range;
    ang_prov.cone_step     = info.cone_step;
    ang_prov.inplane_range = info.inplane_range;
    ang_prov.inplane_step  = info.inplane_step;
    ang_prov.refine_factor = info.refine_factor;
    ang_prov.refine_level  = 0;
    ang_prov.set_symmetry(info.pseudo_sym);
    fprintf(fp," \"angle_table\": [");
    int n_ang = 0;
    M33f R;
    V3f eu;
    for( ang_prov.levels_init(); ang_prov.levels_available(); ang_prov.levels_next() ) {
        for( ang_prov.sym_init(); ang_prov.sym_available(); ang_prov.sym_next() ) {
            for( ang_prov.cone_init(); ang_prov.cone_available(); ang_prov.cone_next() ) {
                for( ang_prov.inplane_init(); ang_prov.inplane_available(); ang_prov.inplane_next() ) {
                    ang_prov.get_current_R(R);
                    Math::Rmat_eZYZ(eu,R);
                    fprintf(fp,"%s\n  [%.9g, %.9g, %.9g]",(n_ang>0)?",":"",eu(0),eu(1),eu(2));
                    n_ang++;
                }
            }
        }
    }
    fprintf(fp,"\n ],\n \"n_angles\": %d,\n \"blocks\": [",n_ang);

    Particle ptcl;
    char f_cc[SUSAN_FILENAME_LENGTH],f_z[SUSAN_FILENAME_LENGTH],f_ang[SUSAN_FILENAME_LENGTH];
    for(uint32 i=0;i<ptcls.n_ptcl;i++) {
        ptcls.get(ptcl,i);
        Tomogram&tomo = tomos.at(tomos.get_cix(ptcl.tomo_id()));
        int r = ptcl.ref_cix();
        V3f pos;
        pos(0) = ptcl.pos().x; pos(1) = ptcl.pos().y; pos(2) = ptcl.pos().z;
        if( info.drift ) {
            pos(0) += ptcl.ali_t[r].x; pos(1) += ptcl.ali_t[r].y; pos(2) += ptcl.ali_t[r].z;
        }
        V3f c = (pos - tomo.tomo_position)/tomo.pix_size + tomo.tomo_center;
        int3 centre = make_int3((int)roundf(c(0)),(int)roundf(c(1)),(int)roundf(c(2)));
        block_file(f_cc ,info.output,ptcl.ptcl_id(),"cc");
        block_file(f_z  ,info.output,ptcl.ptcl_id(),"z");
        block_file(f_ang,info.output,ptcl.ptcl_id(),"ang");
        fprintf(fp,"%s\n  {\"ptcl_id\": %d, \"tomo_id\": %d, \"apix\": %.6g, \"centre_xyz\": [%d, %d, %d], \"corner_xyz\": [%d, %d, %d], ",
                (i>0)?",":"",ptcl.ptcl_id(),ptcl.tomo_id(),tomo.pix_size,centre.x,centre.y,centre.z,
                centre.x+lo.x,centre.y+lo.y,centre.z+lo.z);
        fprintf(fp,"\"shape_xyz\": [%d, %d, %d], \"ali_eu\": [%.9g, %.9g, %.9g], \"ali_t\": [%.9g, %.9g, %.9g], ",
                shape.x,shape.y,shape.z,ptcl.ali_eu[r].x,ptcl.ali_eu[r].y,ptcl.ali_eu[r].z,ptcl.ali_t[r].x,ptcl.ali_t[r].y,ptcl.ali_t[r].z);
        fprintf(fp,"\"files\": {\"cc\": \"%s\", \"z\": \"%s\", \"ang\": \"%s\"}}",f_cc,f_z,f_ang);
    }
    fprintf(fp,"\n ]\n}\n");
    fclose(fp);
}

}

/// Dense output of a 3D block search: best CC, its angle index and the Welford statistics per
/// offset point (on the GPU, updated per orientation), reduced at the end of the block into the
/// cc / z / ang volumes on the voxel grid of the offsets.
class DenseVolumes {

public:
    int   n_pts;
    int   n_vox;
    int3  lo;
    int3  shape;
    int   reduce;
    int   count;

    DenseVolumes(const Vec3*pts,const int in_n_pts,const int in_reduce) {
        n_pts  = in_n_pts;
        reduce = in_reduce;
        count  = 0;
        DenseIO::grid(lo,shape,pts,n_pts);
        n_vox = shape.x*shape.y*shape.z;

        std::vector<int> start(n_vox+1,0),list(n_pts),vox(n_pts);
        for(int i=0;i<n_pts;i++) {
            int x = (int)roundf(pts[i].x)-lo.x, y = (int)roundf(pts[i].y)-lo.y, z = (int)roundf(pts[i].z)-lo.z;
            vox[i] = x + shape.x*(y + shape.y*z);
            start[vox[i]+1]++;
        }
        for(int v=0;v<n_vox;v++) start[v+1] += start[v];
        std::vector<int> fill(start.begin(),start.end()-1);
        for(int i=0;i<n_pts;i++) list[fill[vox[i]]++] = i;

        g_best.alloc(n_pts);
        g_idx.alloc(n_pts);
        g_mean.alloc(n_pts);
        g_m2.alloc(n_pts);
        g_start.alloc(n_vox+1);
        g_list.alloc(n_pts);
        g_vol.alloc(3*(size_t)n_vox);
        c_vol.alloc(3*(size_t)n_vox);
        cudaMemcpy(g_start.ptr,start.data(),sizeof(int)*(n_vox+1),cudaMemcpyHostToDevice);
        cudaMemcpy(g_list.ptr,list.data(),sizeof(int)*n_pts,cudaMemcpyHostToDevice);
    }

    void clear(GPU::Stream&stream) {
        count = 0;
        DenseKernels::clear<<<GPU::div_round_up(n_pts,256),256,0,stream.strm>>>(g_best.ptr,g_idx.ptr,g_mean.ptr,g_m2.ptr,n_pts);
    }

    void update(const float*g_cc,const int ang,GPU::Stream&stream) {
        count++;
        DenseKernels::update<<<GPU::div_round_up(n_pts,256),256,0,stream.strm>>>(g_best.ptr,g_idx.ptr,g_mean.ptr,g_m2.ptr,g_cc,ang,(double)count,n_pts);
    }

    void save(const char*prefix,const uint32 ptcl_id,const int tx,const int ty,const int tz,const float apix,GPU::Stream&stream) {
        DenseKernels::finalize<<<GPU::div_round_up(n_vox,256),256,0,stream.strm>>>(g_vol.ptr,g_best.ptr,g_idx.ptr,g_mean.ptr,g_m2.ptr,
                                                                                    g_start.ptr,g_list.ptr,(double)count,n_vox,reduce);
        GPU::download_async(c_vol.ptr,g_vol.ptr,3*(size_t)n_vox,stream.strm);
        stream.sync();
        int3 corner = make_int3(tx+lo.x,ty+lo.y,tz+lo.z);
        char filename[SUSAN_FILENAME_LENGTH];
        const char*names[3] = {"cc","z","ang"};
        for(int i=0;i<3;i++) {
            DenseIO::block_file(filename,prefix,ptcl_id,names[i]);
            DenseIO::write_block(c_vol.ptr+(size_t)i*n_vox,shape,filename,apix,corner);
        }
    }

protected:
    GPU::GArrSingle  g_best;
    GPU::GArr<int>   g_idx;
    GPU::GArrDouble  g_mean;
    GPU::GArrDouble  g_m2;
    GPU::GArr<int>   g_start;
    GPU::GArr<int>   g_list;
    GPU::GArrSingle  g_vol;
    GPU::GHostSingle c_vol;
};

/// Legacy 2D output (output_type dense, type 2): per projection, the CC map of the orientation
/// with the highest peak, one row per offset point, in {prefix}_{host}[-r{rank}]_{thread}.csv.
class CsvReporter {

public:
    CsvReporter(const Vec3*pts,const int in_n_pts,const int worker_id,const char*prefix) {
        n_pts = in_n_pts;
        p_x.resize(n_pts);
        p_y.resize(n_pts);
        for(int i=0;i<n_pts;i++) {
            p_x[i] = (int)roundf(pts[i].x);
            p_y[i] = (int)roundf(pts[i].y);
        }

        char host[256] = "node";
        gethostname(host,sizeof(host)-1);
        for(char*c=host;*c;c++) {
            if( *c == '.' ) { *c = 0; break; }
            if( !isalnum(*c) && *c != '-' ) *c = '-';
        }
        const char*rank = NULL;
        const char*vars[5] = {"OMPI_COMM_WORLD_RANK","PMI_RANK","PMIX_RANK","MV2_COMM_WORLD_RANK","SLURM_PROCID"};
        for(int i=0;i<5 && rank==NULL;i++) rank = getenv(vars[i]);

        char filename[SUSAN_FILENAME_LENGTH];
        if( rank != NULL )
            snprintf(filename,SUSAN_FILENAME_LENGTH,"%s_%s-r%s_%02d.csv",prefix,host,rank,worker_id);
        else
            snprintf(filename,SUSAN_FILENAME_LENGTH,"%s_%s_%02d.csv",prefix,host,worker_id);
        fp = fopen(filename,"w");
        if( fp == NULL ) {
            fprintf(stderr,"Error creating file %s.\n",filename);
            exit(1);
        }
        fprintf(fp,"TID,PartID,RID,ProjID,ProjW,X,Y,CC\n");
    }

    ~CsvReporter() {
        fclose(fp);
    }

    void save(const int tid,const int rid,const int pid,const int K,const float*prj_w,const float*p_cc) {
        for(int k=0;k<K;k++) {
            for(int i=0;i<n_pts;i++) {
                fprintf(fp,"%d,%d,%d,%d,%f,%d,%d,%.15lf\n",tid,pid,rid,k,prj_w[k],p_x[i],p_y[i],p_cc[k*n_pts+i]);
            }
        }
    }

protected:
    FILE *fp;
    int   n_pts;
    std::vector<int> p_x;
    std::vector<int> p_y;
};

#endif /// DENSE_REPORTER_H
