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

#ifndef ALI_STATS_H
#define ALI_STATS_H

/// 2D alignment statistics (-output_type alignment_stats, type 2), saved to <output stem>.stats.npz:
///   projections/stats (n_ptcl, max_K, NF): PeakStats fields per particle and projection.
///   grouped<gs>/stats, count, label, half: per tomogram, group (gs x gs equal-count bins of the particle x, y) and
///   projection, the statistics of the group's mean CC map at the input in-plane angle; half: peak (x, y) of the
///   A - B split of the group (particle index parity). With independent half-sets (halves = 2) the bins are drawn per
///   half and the groups of half h are (h-1) gs^2 .. h gs^2 - 1 (gold standard: no group mixes the halves).
///   angles (A, 3): rotation vectors (deg, detector frame, left of the projection) of the sequential search: the
///   input orientation, the cone points, the in-plane points (ArgsAli::stats_angles).
///   grouped<gs>/ang_cc, ang_half (gs <= ALI_STATS_ANG_MAX_GS): peak of the group's mean CC map at every angle
///   (n_tomo, G, max_K, A), and of the A and B splits (n_tomo, G, max_K, A, 2).
/// The grouped sections come from partial sums of the raw CC maps (Partial: sum and count per group, projection and
/// half-split), merged by addition: the workers' at the end of each tomogram, and the MPI nodes' for a tomogram split
/// between nodes. A node holds the particle rows of its range (the main node, all of them).

#include <vector>
#include <map>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "peak_stats.h"
#include "npz.h"
#include "stats_groups.h"
#include "particles.h"
#include "tomogram.h"
#include "points_provider.h"
#include "math_cpu.h"

class AliStats : public StatsGroups {

public:
    struct Group {
        int gs;
        int G;                      /// groups: halves x gs x gs
        int offset;                 /// first group of this size in the partial sums
        std::vector<int>    label;  /// (n_ptcl)
        std::vector<int>    count;  /// (n_tomo, G, max_K)
        std::vector<float>  stats;  /// (n_tomo, G, max_K, NF)
        std::vector<float>  half;   /// (n_tomo, G, max_K, 2)
        int                 ang_off;   /// first group of this size in the angle partial sums (-1: no angles)
        std::vector<float>  ang_cc;    /// (n_tomo, G, max_K, A)
        std::vector<float>  ang_half;  /// (n_tomo, G, max_K, A, 2)
    };

    /// Sufficient statistics of the grouped sections of one tomogram: sum of the raw CC maps and count per
    /// (group, projection, half-split).
    struct Partial {
        std::vector<double> sum;    /// (n_groups, max_K, 2, n_pts)
        std::vector<int>    cnt;    /// (n_groups, max_K, 2)
        std::vector<double> asum;   /// (n_ang_groups, max_K, A, 2, n_pts), counts as cnt

        void merge(const Partial&o) {
            for(size_t i=0;i<sum.size();i++) sum[i] += o.sum[i];
            for(size_t i=0;i<cnt.size();i++) cnt[i] += o.cnt[i];
            for(size_t i=0;i<asum.size();i++) asum[i] += o.asum[i];
        }

        void clear() {
            std::fill(sum.begin(),sum.end(),0.0);
            std::fill(cnt.begin(),cnt.end(),0);
            std::fill(asum.begin(),asum.end(),0.0);
        }
    };

    float excl;
    int   max_K;
    int   n_pts;
    int   n_groups;
    int   n_ang;                        /// angles of the sequential search (the input orientation first)
    int   n_ang_groups;
    float step;
    int   lo, hi;                       /// particles processed here: global indices [lo, hi)
    int   row0, n_rows;                 /// particle rows held here
    std::vector<float>   stats;         /// (n_rows, max_K, NF)
    std::vector<float>   apix;
    std::vector<float>   pts_x, pts_y;
    std::vector<Group>   groups;
    std::vector<M33f>    ang_R;         /// (A) rotations of the sequential search
    std::vector<float>   ang_rv;        /// (A, 3) their rotation vectors, deg
    std::vector<int>     done;          /// per tomogram: grouped statistics computed
    std::vector<Partial> work;          /// per worker, current tomogram
    std::map<int,Partial> pending;      /// tomograms only partly processed here (MPI)

    /// Search points, tomograms and group sizes (common to every node).
    void setup(Tomograms&tomos,int off_type,float off_x,float off_y,float off_z,float off_s,
               const std::vector<int>&gs_list,int n_threads,bool ali_halves,const std::vector<M33f>&angles) {
        excl   = 0;   /// data-driven (3 sigma of the main peak)
        setup_tomos(tomos,ali_halves);
        max_K  = tomos.num_proj;
        step   = off_s;
        uint32 np;
        Vec3*pts = PointsProvider::create(np,off_type,off_x,off_y,off_z,off_s);
        n_pts = np;
        pts_x.resize(n_pts);
        pts_y.resize(n_pts);
        for(int j=0;j<n_pts;j++) { pts_x[j] = pts[j].x; pts_y[j] = pts[j].y; }
        delete [] pts;
        apix.resize(n_tomo);
        for(int t=0;t<n_tomo;t++)
            apix[t] = tomos[t].pix_size;
        ang_R = angles;
        n_ang = ang_R.size();
        ang_rv.resize(3*n_ang);
        for(int a=0;a<n_ang;a++) rotation_vector(ang_rv.data()+3*a,ang_R[a]);
        groups.resize(gs_list.size());
        n_groups = 0;
        n_ang_groups = 0;
        for(size_t q=0;q<gs_list.size();q++) {
            Group&g = groups[q];
            g.gs = gs_list[q];
            g.G  = halves*g.gs*g.gs;
            int G = g.G;
            g.offset = n_groups;
            n_groups += G;
            g.count.assign((size_t)n_tomo*G*max_K,0);
            g.stats.assign((size_t)n_tomo*G*max_K*PeakStats::NF,NAN);
            g.half.assign((size_t)n_tomo*G*max_K*2,NAN);
            g.ang_off = -1;
            if( n_ang > 1 && g.gs <= ALI_STATS_ANG_MAX_GS ) {
                g.ang_off = n_ang_groups;
                n_ang_groups += G;
                g.ang_cc.assign((size_t)n_tomo*G*max_K*n_ang,NAN);
                g.ang_half.assign((size_t)n_tomo*G*max_K*n_ang*2,NAN);
            }
        }
        work.resize(n_threads);
        for(Partial&w: work) {
            w.sum.assign((size_t)n_groups*max_K*2*n_pts,0.0);
            w.cnt.assign((size_t)n_groups*max_K*2,0);
            w.asum.assign((size_t)n_ang_groups*max_K*n_ang*2*n_pts,0.0);
        }
        done.assign(n_tomo,0);
        row.resize(n_pts);
        tmp.resize(PeakStats::NF);
    }

    /// Tomogram index and group labels of every particle, from the full particle list.
    void set_layout(Particles&ptcls,Tomograms&tomos) {
        std::vector<float> px, py;
        std::vector<int>   hid;
        read_particles(ptcls,tomos,hid,px,py);
        for(Group&g: groups)
            label_groups(g.label,g.gs,hid,px,py);
        set_tomo_ranges();
    }

    /// Particles processed here and particle rows held here (single node: 0, n_ptcl, 0, n_ptcl).
    void set_range(int in_lo,int in_hi,int in_row0,int in_n_rows) {
        lo = in_lo;
        hi = in_hi;
        row0 = in_row0;
        n_rows = in_n_rows;
        stats.assign((size_t)n_rows*max_K*PeakStats::NF,NAN);
    }

    float*particle_row(int gix,int k) {
        return stats.data() + ((size_t)(gix-row0)*max_K + k)*PeakStats::NF;
    }

    /// worker `thr` adds the raw CC row (not weighted) at the input in-plane angle of particle gix, projection k
    void add_group_row(int thr,int gix,int k,const float*cc_row) {
        Partial&w = work[thr];
        for(Group&g: groups) {
            int l = g.label[gix];
            if( l < 0 ) continue;
            size_t base = ((size_t)(g.offset + l)*max_K + k)*2 + (gix & 1);       /// halves: particle index parity
            double*a = w.sum.data() + base*n_pts;
            for(int j=0;j<n_pts;j++) a[j] += cc_row[j];
            w.cnt[base]++;
        }
    }

    /// worker `thr` adds the raw CC row of angle a (sequential search) of particle gix, projection k
    void add_group_ang(int thr,int gix,int k,int a,const float*cc_row) {
        Partial&w = work[thr];
        for(Group&g: groups) {
            int l = g.label[gix];
            if( l < 0 || g.ang_off < 0 ) continue;
            size_t base = (((size_t)(g.ang_off + l)*max_K + k)*n_ang + a)*2 + (gix & 1);
            double*s = w.asum.data() + base*n_pts;
            for(int j=0;j<n_pts;j++) s[j] += cc_row[j];
        }
    }

    /// End of tomogram t, all workers idle: merges their partial sums; the grouped statistics when the whole tomogram
    /// was processed here, otherwise kept for the merge with the other nodes.
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

    /// Grouped statistics of the merged partial sums (main node, after merge_node of every node).
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
        for(const Group&g: groups) put_vec(b,g.label);
        return b;
    }

    void unpack_layout(const std::vector<uint8_t>&b) {
        size_t pos = 0;
        get(b,pos,n_ptcl);
        get_vec(b,pos,tix,n_ptcl);
        for(Group&g: groups) get_vec(b,pos,g.label,n_ptcl);
        set_tomo_ranges();
    }

    /// MPI: what a node sends to the main node: its particle rows, the grouped statistics of the tomograms it
    /// completed and the partial sums of the others.
    std::vector<uint8_t> pack_node() const {
        std::vector<uint8_t> b;
        put(b,row0);
        put(b,n_rows);
        put_vec(b,stats);
        int nd = 0;
        for(int t=0;t<n_tomo;t++) nd += done[t];
        put(b,nd);
        for(int t=0;t<n_tomo;t++) {
            if( !done[t] ) continue;
            put(b,t);
            for(const Group&g: groups) {
                size_t n = (size_t)g.G*max_K;
                put_raw(b,g.count.data()+t*n,n*sizeof(int));
                put_raw(b,g.stats.data()+t*n*PeakStats::NF,n*PeakStats::NF*sizeof(float));
                put_raw(b,g.half.data()+t*n*2,n*2*sizeof(float));
                if( g.ang_off >= 0 ) {
                    put_raw(b,g.ang_cc.data()+t*n*n_ang,n*n_ang*sizeof(float));
                    put_raw(b,g.ang_half.data()+t*n*n_ang*2,n*n_ang*2*sizeof(float));
                }
            }
        }
        put(b,(int)pending.size());
        for(const auto&it: pending) {
            put(b,it.first);
            put_vec(b,it.second.sum);
            put_vec(b,it.second.cnt);
            put_vec(b,it.second.asum);
        }
        return b;
    }

    /// MPI, main node: adds a node's pack_node (rows into place, completed tomograms copied, partial sums merged).
    void merge_node(const std::vector<uint8_t>&b) {
        size_t pos = 0;
        int r0, nr, nd, np;
        get(b,pos,r0);
        get(b,pos,nr);
        size_t nf = (size_t)nr*max_K*PeakStats::NF;
        get_raw(b,pos,stats.data()+((size_t)(r0-row0)*max_K)*PeakStats::NF,nf*sizeof(float));
        get(b,pos,nd);
        for(int i=0;i<nd;i++) {
            int t;
            get(b,pos,t);
            for(Group&g: groups) {
                size_t n = (size_t)g.G*max_K;
                get_raw(b,pos,g.count.data()+t*n,n*sizeof(int));
                get_raw(b,pos,g.stats.data()+t*n*PeakStats::NF,n*PeakStats::NF*sizeof(float));
                get_raw(b,pos,g.half.data()+t*n*2,n*2*sizeof(float));
                if( g.ang_off >= 0 ) {
                    get_raw(b,pos,g.ang_cc.data()+t*n*n_ang,n*n_ang*sizeof(float));
                    get_raw(b,pos,g.ang_half.data()+t*n*n_ang*2,n*n_ang*2*sizeof(float));
                }
            }
            done[t] = 1;
        }
        get(b,pos,np);
        Partial p;
        for(int i=0;i<np;i++) {
            int t;
            get(b,pos,t);
            get_vec(b,pos,p.sum,(size_t)n_groups*max_K*2*n_pts);
            get_vec(b,pos,p.cnt,(size_t)n_groups*max_K*2);
            get_vec(b,pos,p.asum,(size_t)n_ang_groups*max_K*n_ang*2*n_pts);
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
        return s + ".stats.npz";
    }

    bool save(const char*output) {
        using Npz::npy;
        const int NF = PeakStats::NF;
        std::vector<Npz::Entry> ents;
        ents.push_back(npy("projections/stats","<f4",{(size_t)n_ptcl,(size_t)max_K,(size_t)NF},stats.data(),stats.size()*sizeof(float)));
        std::vector<uint32_t> uf(NF*8,0);
        for(int f=0;f<NF;f++)
            for(int c=0;c<8 && PeakStats::FIELDS[f][c];c++) uf[f*8+c] = (uint32_t)PeakStats::FIELDS[f][c];
        ents.push_back(npy("projections/fields","<U8",{(size_t)NF},uf.data(),uf.size()*sizeof(uint32_t)));
        ents.push_back(npy("tomo_id","<i4",{(size_t)n_tomo},tomo_id.data(),tomo_id.size()*sizeof(int)));
        ents.push_back(npy("off_step","<f4",{},&step,sizeof(float)));
        ents.push_back(npy("apix","<f4",{(size_t)n_tomo},apix.data(),apix.size()*sizeof(float)));
        ents.push_back(npy("halves","<i4",{},&halves,sizeof(int)));
        ents.push_back(npy("angles","<f4",{(size_t)n_ang,3},ang_rv.data(),ang_rv.size()*sizeof(float)));
        for(Group&g: groups) {
            std::string pfx = "grouped" + std::to_string(g.gs) + "/";
            size_t G = (size_t)g.G;
            ents.push_back(npy(pfx+"stats","<f4",{(size_t)n_tomo,G,(size_t)max_K,(size_t)NF},g.stats.data(),g.stats.size()*sizeof(float)));
            ents.push_back(npy(pfx+"count","<i4",{(size_t)n_tomo,G,(size_t)max_K},g.count.data(),g.count.size()*sizeof(int)));
            ents.push_back(npy(pfx+"label","<i4",{(size_t)n_ptcl},g.label.data(),g.label.size()*sizeof(int)));
            ents.push_back(npy(pfx+"half","<f4",{(size_t)n_tomo,G,(size_t)max_K,2},g.half.data(),g.half.size()*sizeof(float)));
            if( g.ang_off >= 0 ) {
                ents.push_back(npy(pfx+"ang_cc","<f4",{(size_t)n_tomo,G,(size_t)max_K,(size_t)n_ang},g.ang_cc.data(),g.ang_cc.size()*sizeof(float)));
                ents.push_back(npy(pfx+"ang_half","<f4",{(size_t)n_tomo,G,(size_t)max_K,(size_t)n_ang,2},g.ang_half.data(),g.ang_half.size()*sizeof(float)));
            }
        }
        return Npz::write(stats_path(output).c_str(),ents);
    }

protected:
    PeakStats::Workspace ws;
    std::vector<float>   row;
    std::vector<float>   tmp;

    static void rotation_vector(float*rv,const M33f&R) {
        double c = std::min(std::max((R.trace()-1.0)/2.0,-1.0),1.0);
        double th = acos(c);
        double f = (th < 1e-9) ? 0.5 : th/(2*sin(th));
        rv[0] = (float)(f*(R(2,1)-R(1,2))*180.0/M_PI);
        rv[1] = (float)(f*(R(0,2)-R(2,0))*180.0/M_PI);
        rv[2] = (float)(f*(R(1,0)-R(0,1))*180.0/M_PI);
    }

    /// Grouped statistics of tomogram t from its partial sums.
    void finalize(int t,const Partial&p) {
        for(Group&g: groups) {
            int G = g.G;
            for(int l=0;l<G;l++)
                for(int k=0;k<max_K;k++) {
                    size_t base = ((size_t)(g.offset + l)*max_K + k)*2;
                    int ch[2] = {p.cnt[base],p.cnt[base+1]};
                    const double*s = p.sum.data() + base*n_pts;   /// half 0, then half 1
                    size_t out = ((size_t)t*G + l)*max_K + k;
                    int c = ch[0] + ch[1];
                    g.count[out] = c;
                    if( c == 0 ) continue;
                    for(int j=0;j<n_pts;j++) row[j] = (float)((s[j]+s[n_pts+j])/c);
                    PeakStats::compute(ws,g.stats.data()+out*PeakStats::NF,row.data(),n_pts,pts_x.data(),pts_y.data(),step,excl);
                    if( ch[0] > 0 && ch[1] > 0 ) {
                        float pk[2][2];
                        for(int h=0;h<2;h++) {
                            for(int j=0;j<n_pts;j++) row[j] = (float)(s[h*n_pts+j]/ch[h]);
                            PeakStats::compute(ws,tmp.data(),row.data(),n_pts,pts_x.data(),pts_y.data(),step,excl);
                            pk[h][0] = tmp[1]; pk[h][1] = tmp[2];
                        }
                        g.half[out*2+0] = pk[0][0] - pk[1][0];
                        g.half[out*2+1] = pk[0][1] - pk[1][1];
                    }
                    if( g.ang_off < 0 ) continue;
                    for(int a=0;a<n_ang;a++) {
                        const double*sa = p.asum.data() + ((((size_t)(g.ang_off + l)*max_K + k)*n_ang + a)*2)*n_pts;
                        for(int j=0;j<n_pts;j++) row[j] = (float)((sa[j]+sa[n_pts+j])/c);
                        PeakStats::compute(ws,tmp.data(),row.data(),n_pts,pts_x.data(),pts_y.data(),step,excl);
                        g.ang_cc[out*n_ang+a] = tmp[0];
                        for(int h=0;h<2;h++) {
                            if( ch[h] == 0 ) continue;
                            for(int j=0;j<n_pts;j++) row[j] = (float)(sa[h*n_pts+j]/ch[h]);
                            PeakStats::compute(ws,tmp.data(),row.data(),n_pts,pts_x.data(),pts_y.data(),step,excl);
                            g.ang_half[(out*n_ang+a)*2+h] = tmp[0];
                        }
                    }
                }
        }
        done[t] = 1;
    }
};

#endif /// ALI_STATS_H
