/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
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

#ifndef CTF_STATS_H
#define CTF_STATS_H

/// CTF statistics (-output_type refinement_stats), saved to <output stem>.ctfstats.npz. Per particle and projection, over 
/// the linearized coordinate v = s^2 D(phi) / D_ref (D(phi) the particle's metadata defocus, D_ref the median metadata 
/// defocus of the projection over the tomogram's particles; NB bins of width vmax / NB), with G = F P exp(2 pi i k.t / NP),
/// (F substack, P reference projection without CTF, t the sub-pixel shift where the CC is read; F and P whitened as cc_type):
///   group<gs>/x, w, s, c (n_tomo, G, 2, max_K, NB): sum Re(G), |P|^2, |F|^2, pixel count of the groups (gs x gs
///   equal-count bins of the particle x, y in each tomogram; the groups of half h are (h-1) gs^2 .. h gs^2 - 1 with
///   independent half-sets, halves = 2) split by the particle index parity; group<gs>/label (n_ptcl).
///   shift/ix, iy, wxx, wyy, wxy (n_tomo, halves, 2, max_K, NB): sum Im(G) kx, Im(G) ky, kx^2 |P|^2, ky^2 |P|^2,
///   kx ky |P|^2 of all the particles of the half-set and parity (kx, ky in pixels of the padded box NP).
///   curves (n_ptcl, max_K, ND): max(sum_v c X, 0) / sqrt(sum_v c^2 W) of the particle's own profile, c the CTF at
///   D_ref + delta; delta (ND), dref (n_tomo, max_K), vmax, nb, box (NP), apix, kv, cs, ac, tomo_id, halves.
/// The sums come from partial sums (Partial: doubles per worker), merged by addition: the workers' at the end of each
/// tomogram, and the MPI nodes' for a tomogram split between nodes. A node holds the curve rows of its range (the main
/// node, all of them).

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include "datatypes.h"
#include "npz.h"
#include "stats_groups.h"
#include "particles.h"
#include "tomogram.h"
#include "gpu_kernel.h"
#include "gpu_kernel_ctf.h"

namespace GpuKernelsCtfStats {

__global__ void profile(float*p_prof,const float2*p_ss,const float2*p_prj,const Proj2D*p_ali,const Defocus*p_def,
                        const float*p_dref,const CtfConst ctf,const float vmax,const int NB,const float r_lo,const float r_hi,
                        const int3 ss_siz) {
    int3 ss_idx = get_th_idx();
    if( ss_idx.x < ss_siz.x && ss_idx.y < ss_siz.y && ss_idx.z < ss_siz.z ) {
        int k = ss_idx.z;
        if( p_ali[k].w <= 0 || p_dref[k] <= 0 ) return;
        float x = ss_idx.x;
        float y = ss_idx.y - ss_siz.y/2;
        float R = sqrtf(x*x + y*y);
        if( R < r_lo || R > r_hi ) return;
        float s = R/(ss_siz.y*ctf.apix);
        float v = s*s*GpuKernelsCtf::calc_def(x,y,p_def[k])/p_dref[k];
        int b = (int)(v/vmax*NB);
        if( b < 0 || b >= NB ) return;
        long idx = get_3d_idx(ss_idx,ss_siz);
        float2 F = p_ss[idx];
        float2 P = p_prj[idx];
        float2 G = cuCmulf(F,P);
        float ph = 2*M_PI*(x*p_ali[k].t.x + y*p_ali[k].t.y)/ss_siz.y;
        float c = cosf(ph), sn = sinf(ph);
        float re = G.x*c - G.y*sn;
        float im = G.x*sn + G.y*c;
        float P2 = P.x*P.x + P.y*P.y;
        size_t Q = (size_t)ss_siz.z*NB;
        float*p = p_prof + (size_t)k*NB + b;
        atomicAdd(p      ,re);
        atomicAdd(p +   Q,P2);
        atomicAdd(p + 2*Q,F.x*F.x + F.y*F.y);
        atomicAdd(p + 3*Q,1.0f);
        atomicAdd(p + 4*Q,im*x);
        atomicAdd(p + 5*Q,im*y);
        atomicAdd(p + 6*Q,x*x*P2);
        atomicAdd(p + 7*Q,y*y*P2);
        atomicAdd(p + 8*Q,x*y*P2);
    }
}

__global__ void curves(float*p_out,const float*p_prof,const float*p_dref,const CtfConst ctf,const float vmax,const int NB,
                       const float d0,const float dstep,const int ND,const int K) {
    int3 ss_idx = get_th_idx();
    if( ss_idx.x < ND && ss_idx.y < K && ss_idx.z < 1 ) {
        int j = ss_idx.x, k = ss_idx.y;
        float out = NAN;
        if( p_dref[k] > 0 ) {
            float D = p_dref[k] + d0 + j*dstep;
            const float*X = p_prof + (size_t)k*NB;
            const float*W = p_prof + (size_t)K*NB + (size_t)k*NB;
            float num = 0, den = 0;
            for(int b=0;b<NB;b++) {
                if( W[b] <= 0 ) continue;
                float v = (b + 0.5f)*vmax/NB;
                float g = GpuKernelsCtf::calc_gamma(D,ctf.LambdaPi,ctf.CsLambda3PiH,v);
                float cc = GpuKernelsCtf::calc_ctf(g,ctf.AC,ctf.CA);
                num += cc*X[b];
                den += cc*cc*W[b];
            }
            if( den > 0 ) out = fmaxf(num,0.0f)/sqrtf(den);
        }
        p_out[(size_t)k*ND + j] = out;
    }
}

}

class CtfStats : public StatsGroups {

public:
    static const int NQ  = 4;   /// group quantities: Re(G), |P|^2, |F|^2, count
    static const int NS  = 5;   /// shift quantities
    static const int NPQ = 9;   /// per-particle profile quantities

    struct Group {
        int gs;
        int G;                      /// groups: halves x gs x gs
        int offset;                 /// first group of this size in the partial sums
        std::vector<int>    label;  /// (n_ptcl)
        std::vector<double> sum[NQ];   /// (n_tomo, G, 2, max_K, NB)
    };

    /// Sufficient statistics of one tomogram, by quantity: the groups and the shift sums, split by particle parity.
    struct Partial {
        std::vector<double> g;      /// (NQ, n_groups, 2, max_K, NB)
        std::vector<double> s;      /// (NS, halves, 2, max_K, NB)

        void merge(const Partial&o) {
            for(size_t i=0;i<g.size();i++) g[i] += o.g[i];
            for(size_t i=0;i<s.size();i++) s[i] += o.s[i];
        }

        void clear() {
            std::fill(g.begin(),g.end(),0.0);
            std::fill(s.begin(),s.end(),0.0);
        }
    };

    int   max_K;
    int   NB;
    int   ND;
    int   NP;
    float vmax;
    float d0;
    float dstep;
    int   n_groups;
    int   lo, hi;                       /// particles processed here: global indices [lo, hi)
    int   row0, n_rows;                 /// curve rows held here
    std::vector<float>   apix, kv, cs, ac;
    std::vector<int>     half;          /// half-set (0, 1) of each particle
    std::vector<float>   dref;          /// (n_tomo, max_K)
    std::vector<float>   delta;         /// (ND)
    std::vector<float>   curves;        /// (n_rows, max_K, ND)
    std::vector<double>  shift[NS];     /// (n_tomo, halves, 2, max_K, NB)
    std::vector<Group>   groups;
    std::vector<int>     done;          /// per tomogram: statistics computed
    std::vector<Partial> work;          /// per worker, current tomogram
    std::map<int,Partial> pending;      /// tomograms only partly processed here (MPI)

    /// Profile bins, defocus grid (the refiner's: -def_range/2 .. def_range/2), tomograms and group sizes (common to every node).
    void setup(Tomograms&tomos,const std::vector<int>&gs_list,int n_threads,bool use_halves,int box_pad,float fpix_max,int box,
               float def_range,float def_step) {
        setup_tomos(tomos,use_halves);
        max_K = tomos.num_proj;
        NB    = SUSAN_CTF_STATS_BINS;
        NP    = box_pad;
        int nd = (int)fmax(round(0.5f*def_range/def_step),1.0);
        ND    = 2*nd + 1;
        dstep = 0.5f*def_range/nd;
        d0    = -nd*dstep;
        delta.resize(ND);
        for(int j=0;j<ND;j++) delta[j] = d0 + j*dstep;
        apix.resize(n_tomo); kv.resize(n_tomo); cs.resize(n_tomo); ac.resize(n_tomo);
        float amin = 1e9;
        for(int t=0;t<n_tomo;t++) {
            apix[t] = tomos[t].pix_size; kv[t] = tomos[t].KV; cs[t] = tomos[t].CS; ac[t] = tomos[t].AC;
            amin = fmin(amin,apix[t]);
        }
        float smax = fpix_max/(box*amin);
        vmax = SUSAN_CTF_STATS_VMAX*smax*smax;
        groups.resize(gs_list.size());
        n_groups = 0;
        for(size_t q=0;q<gs_list.size();q++) {
            Group&g = groups[q];
            g.gs = gs_list[q];
            g.G  = halves*g.gs*g.gs;
            g.offset = n_groups;
            n_groups += g.G;
            for(int c=0;c<NQ;c++)
                g.sum[c].assign((size_t)n_tomo*g.G*2*max_K*NB,0.0);
        }
        for(int c=0;c<NS;c++)
            shift[c].assign((size_t)n_tomo*halves*2*max_K*NB,0.0);
        work.resize(n_threads);
        for(Partial&w: work) {
            w.g.assign((size_t)NQ*n_groups*2*max_K*NB,0.0);
            w.s.assign((size_t)NS*halves*2*max_K*NB,0.0);
        }
        done.assign(n_tomo,0);
    }

    /// Tomogram index, group labels and reference defocus of every particle, from the full particle list.
    void set_layout(Particles&ptcls,Tomograms&tomos) {
        std::vector<float> px, py;
        std::vector<int>   hid;
        read_particles(ptcls,tomos,hid,px,py);
        half.assign(n_ptcl,0);
        std::vector< std::vector<float> > dk((size_t)n_tomo*max_K);
        Particle p;
        for(int i=0;i<n_ptcl;i++) {
            if( halves == 2 ) half[i] = hid[i] - 1;
            if( tix[i] < 0 ) continue;
            ptcls.get(p,i);
            for(int k=0;k<(int)tomos[tix[i]].num_proj;k++)
                if( p.prj_w[k] > 0 ) dk[(size_t)tix[i]*max_K + k].push_back(0.5f*(p.def[k].U + p.def[k].V));
        }
        dref.assign((size_t)n_tomo*max_K,0);
        for(size_t q=0;q<dk.size();q++) {
            if( dk[q].empty() ) continue;
            std::nth_element(dk[q].begin(),dk[q].begin()+dk[q].size()/2,dk[q].end());
            dref[q] = dk[q][dk[q].size()/2];
        }
        for(Group&g: groups)
            label_groups(g.label,g.gs,hid,px,py);
        set_tomo_ranges();
    }

    /// Particles processed here and curve rows held here (single node: 0, n_ptcl, 0, n_ptcl).
    void set_range(int in_lo,int in_hi,int in_row0,int in_n_rows) {
        lo = in_lo;
        hi = in_hi;
        row0 = in_row0;
        n_rows = in_n_rows;
        curves.assign((size_t)n_rows*max_K*ND,NAN);
    }

    const float*dref_tomo(int gix) const {
        return dref.data() + (size_t)tix[gix]*max_K;
    }

    float*curve_row(int gix) {
        return curves.data() + (size_t)(gix-row0)*max_K*ND;
    }

    /// worker `thr` adds the profiles (NPQ, K, NB) of particle gix
    void add_particle(int thr,int gix,int K,const float*prof) {
        if( tix[gix] < 0 ) return;
        int par = gix & 1;       /// halves: particle index parity
        size_t Q = (size_t)K*NB;
        Partial&w = work[thr];
        for(Group&g: groups) {
            int l = g.label[gix];
            if( l < 0 ) continue;
            for(int c=0;c<NQ;c++) {
                double*a = w.g.data() + (((size_t)c*n_groups + g.offset + l)*2 + par)*max_K*NB;
                for(size_t kb=0;kb<Q;kb++) a[kb] += prof[c*Q + kb];
            }
        }
        for(int c=0;c<NS;c++) {
            double*a = w.s.data() + (((size_t)c*halves + half[gix])*2 + par)*max_K*NB;
            for(size_t kb=0;kb<Q;kb++) a[kb] += prof[(NQ + c)*Q + kb];
        }
    }

    /// End of tomogram t, all workers idle: merges their partial sums; the statistics when the whole tomogram was
    /// processed here, otherwise kept for the merge with the other nodes.
    void end_tomo(int t) {
        if( t < 0 || groups.empty() ) return;
        Partial&m = work[0];
        for(size_t w=1;w<work.size();w++) m.merge(work[w]);
        if( first[t] >= lo && last[t] < hi ) {
            finalize(t,m);
        }
        else {
            auto it = pending.find(t);
            if( it == pending.end() )
                pending[t] = m;
            else
                it->second.merge(m);
        }
        for(Partial&w: work) w.clear();
    }

    /// Statistics of the merged partial sums (main node, after merge_node of every node).
    void finish() {
        for(auto&it: pending)
            finalize(it.first,it.second);
        pending.clear();
    }

    /// MPI: the layout (main node -> nodes).
    std::vector<uint8_t> pack_layout() const {
        std::vector<uint8_t> b;
        put(b,n_ptcl);
        put_vec(b,tix);
        put_vec(b,half);
        put_vec(b,dref);
        for(const Group&g: groups) put_vec(b,g.label);
        return b;
    }

    void unpack_layout(const std::vector<uint8_t>&b) {
        size_t pos = 0;
        get(b,pos,n_ptcl);
        get_vec(b,pos,tix,n_ptcl);
        get_vec(b,pos,half,n_ptcl);
        get_vec(b,pos,dref,(size_t)n_tomo*max_K);
        for(Group&g: groups) get_vec(b,pos,g.label,n_ptcl);
        set_tomo_ranges();
    }

    /// MPI: what a node sends to the main node: its curve rows, the statistics of the tomograms it completed and the
    /// partial sums of the others.
    std::vector<uint8_t> pack_node() const {
        std::vector<uint8_t> b;
        put(b,row0);
        put(b,n_rows);
        put_vec(b,curves);
        int nd = 0;
        for(int t=0;t<n_tomo;t++) nd += done[t];
        put(b,nd);
        for(int t=0;t<n_tomo;t++) {
            if( !done[t] ) continue;
            put(b,t);
            for(const Group&g: groups) {
                size_t n = (size_t)g.G*2*max_K*NB;
                for(int c=0;c<NQ;c++) put_raw(b,g.sum[c].data()+t*n,n*sizeof(double));
            }
            size_t ns = (size_t)halves*2*max_K*NB;
            for(int c=0;c<NS;c++) put_raw(b,shift[c].data()+t*ns,ns*sizeof(double));
        }
        put(b,(int)pending.size());
        for(const auto&it: pending) {
            put(b,it.first);
            put_vec(b,it.second.g);
            put_vec(b,it.second.s);
        }
        return b;
    }

    /// MPI, main node: adds a node's pack_node (rows into place, completed tomograms copied, partial sums merged).
    void merge_node(const std::vector<uint8_t>&b) {
        size_t pos = 0;
        int r0, nr, nd, np;
        get(b,pos,r0);
        get(b,pos,nr);
        get_raw(b,pos,curves.data()+(size_t)(r0-row0)*max_K*ND,(size_t)nr*max_K*ND*sizeof(float));
        get(b,pos,nd);
        for(int i=0;i<nd;i++) {
            int t;
            get(b,pos,t);
            for(Group&g: groups) {
                size_t n = (size_t)g.G*2*max_K*NB;
                for(int c=0;c<NQ;c++) get_raw(b,pos,g.sum[c].data()+t*n,n*sizeof(double));
            }
            size_t ns = (size_t)halves*2*max_K*NB;
            for(int c=0;c<NS;c++) get_raw(b,pos,shift[c].data()+t*ns,ns*sizeof(double));
            done[t] = 1;
        }
        get(b,pos,np);
        Partial p;
        for(int i=0;i<np;i++) {
            int t;
            get(b,pos,t);
            get_vec(b,pos,p.g,(size_t)NQ*n_groups*2*max_K*NB);
            get_vec(b,pos,p.s,(size_t)NS*halves*2*max_K*NB);
            auto it = pending.find(t);
            if( it == pending.end() )
                pending[t] = p;
            else
                it->second.merge(p);
        }
    }

    static std::string stats_path(const char*ptcls_out) {
        std::string s(ptcls_out);
        const std::string ext = ".ptclsraw";
        if( s.size() >= ext.size() && s.compare(s.size()-ext.size(),ext.size(),ext) == 0 )
            s = s.substr(0,s.size()-ext.size());
        return s + ".ctfstats.npz";
    }

    bool save(const char*output) {
        using Npz::npy;
        std::vector<Npz::Entry> ents;
        const char*qn[NQ] = {"x","w","s","c"};
        for(Group&g: groups) {
            std::string pfx = "group" + std::to_string(g.gs) + "/";
            for(int c=0;c<NQ;c++)
                ents.push_back(npy(pfx+qn[c],"<f8",{(size_t)n_tomo,(size_t)g.G,2,(size_t)max_K,(size_t)NB},g.sum[c].data(),g.sum[c].size()*sizeof(double)));
            ents.push_back(npy(pfx+"label","<i4",{(size_t)n_ptcl},g.label.data(),g.label.size()*sizeof(int)));
        }
        const char*sn[NS] = {"ix","iy","wxx","wyy","wxy"};
        for(int c=0;c<NS;c++)
            ents.push_back(npy(std::string("shift/")+sn[c],"<f8",{(size_t)n_tomo,(size_t)halves,2,(size_t)max_K,(size_t)NB},shift[c].data(),shift[c].size()*sizeof(double)));
        ents.push_back(npy("curves","<f4",{(size_t)n_ptcl,(size_t)max_K,(size_t)ND},curves.data(),curves.size()*sizeof(float)));
        ents.push_back(npy("delta","<f4",{(size_t)ND},delta.data(),delta.size()*sizeof(float)));
        ents.push_back(npy("dref","<f4",{(size_t)n_tomo,(size_t)max_K},dref.data(),dref.size()*sizeof(float)));
        ents.push_back(npy("vmax","<f4",{},&vmax,sizeof(float)));
        ents.push_back(npy("nb","<i4",{},&NB,sizeof(int)));
        ents.push_back(npy("box","<i4",{},&NP,sizeof(int)));
        ents.push_back(npy("tomo_id","<i4",{(size_t)n_tomo},tomo_id.data(),tomo_id.size()*sizeof(int)));
        ents.push_back(npy("apix","<f4",{(size_t)n_tomo},apix.data(),apix.size()*sizeof(float)));
        ents.push_back(npy("kv","<f4",{(size_t)n_tomo},kv.data(),kv.size()*sizeof(float)));
        ents.push_back(npy("cs","<f4",{(size_t)n_tomo},cs.data(),cs.size()*sizeof(float)));
        ents.push_back(npy("ac","<f4",{(size_t)n_tomo},ac.data(),ac.size()*sizeof(float)));
        ents.push_back(npy("halves","<i4",{},&halves,sizeof(int)));
        return Npz::write(stats_path(output).c_str(),ents);
    }

protected:
    /// Statistics of tomogram t from its partial sums.
    void finalize(int t,const Partial&p) {
        for(Group&g: groups) {
            size_t n = (size_t)g.G*2*max_K*NB;
            for(int c=0;c<NQ;c++)
                memcpy(g.sum[c].data()+t*n,p.g.data()+((size_t)c*n_groups + g.offset)*2*max_K*NB,n*sizeof(double));
        }
        size_t ns = (size_t)halves*2*max_K*NB;
        for(int c=0;c<NS;c++)
            memcpy(shift[c].data()+t*ns,p.s.data()+c*ns,ns*sizeof(double));
        done[t] = 1;
    }
};

#endif /// CTF_STATS_H
