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

#ifndef STATS_GROUPS_H
#define STATS_GROUPS_H

/// Particle groups shared by the statistics outputs (AliStats, CtfStats): the tomogram of each particle, its range, and
/// the equal-count gs x gs bins of the particle x, y in each tomogram (per half-set when halves = 2), plus the byte
/// packing of the MPI transfers.

#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "particles.h"
#include "tomogram.h"

class StatsGroups {

public:
    int   n_ptcl;
    int   n_tomo;
    int   halves;                       /// 1, or 2 with independent half-sets (groups per half)
    std::vector<int>     tix;           /// tomogram index of each particle (-1: none)
    std::vector<int>     first, last;   /// global particle range of each tomogram (-1: no particles)
    std::vector<int>     tomo_id;

    int tomo_index(int tid) const {
        for(int t=0;t<n_tomo;t++) if( tomo_id[t] == tid ) return t;
        return -1;
    }

protected:
    void setup_tomos(Tomograms&tomos,bool ali_halves) {
        halves = ali_halves ? 2 : 1;
        n_tomo = tomos.num_tomo;
        tomo_id.resize(n_tomo);
        for(int t=0;t<n_tomo;t++) tomo_id[t] = tomos[t].tomo_id;
    }

    /// Tomogram index, half-set, and position (including the alignment shift) of every particle.
    void read_particles(Particles&ptcls,Tomograms&tomos,std::vector<int>&hid,std::vector<float>&px,std::vector<float>&py) {
        n_ptcl = ptcls.n_ptcl;
        tix.assign(n_ptcl,-1);
        px.resize(n_ptcl);
        py.resize(n_ptcl);
        hid.resize(n_ptcl);
        Particle p;
        for(int i=0;i<n_ptcl;i++) {
            ptcls.get(p,i);
            tix[i] = tomos.get_cix(p.tomo_id());
            px[i]  = p.pos().x + p.ali_t[0].x;
            py[i]  = p.pos().y + p.ali_t[0].y;
            hid[i] = p.half_id();
        }
    }

    void label_groups(std::vector<int>&label,int gs,const std::vector<int>&hid,const std::vector<float>&px,const std::vector<float>&py) {
        label.assign(n_ptcl,-1);
        for(int t=0;t<n_tomo;t++)
            for(int h=1;h<=halves;h++)
                label_tomo(label,gs,t,h,hid,px,py);
    }

    void set_tomo_ranges() {
        first.assign(n_tomo,-1);
        last.assign(n_tomo,-1);
        for(int i=0;i<n_ptcl;i++) {
            int t = tix[i];
            if( t < 0 ) continue;
            if( first[t] < 0 ) first[t] = i;
            last[t] = i;
        }
    }

    /// equal-count gs x gs bins of the particle x, y of tomogram t (as numpy quantile + searchsorted(side='right') - 1);
    /// with halves = 2, of the particles of half h only, labels offset by (h-1) gs^2
    void label_tomo(std::vector<int>&label,int gs,int t,int h,const std::vector<int>&hid,const std::vector<float>&px,const std::vector<float>&py) {
        std::vector<int> idx;
        for(int i=0;i<n_ptcl;i++) if( tix[i] == t && (halves == 1 || hid[i] == h) ) idx.push_back(i);
        if( idx.empty() ) return;
        auto edges = [&](const std::vector<float>&v) {
            std::vector<double> s;
            for(int i: idx) s.push_back(v[i]);
            std::sort(s.begin(),s.end());
            std::vector<double> e(gs+1);
            for(int j=0;j<=gs;j++) {
                double q = (double)j*(s.size()-1)/gs;
                size_t lo = (size_t)floor(q);
                size_t hi = std::min(lo+1,s.size()-1);
                e[j] = s[lo] + (q-lo)*(s[hi]-s[lo]);
            }
            return e;
        };
        std::vector<double> ex = edges(px), ey = edges(py);
        auto bin = [&](const std::vector<double>&e,double v) {
            int b = (int)(std::upper_bound(e.begin(),e.end(),v) - e.begin()) - 1;
            return std::min(std::max(b,0),gs-1);
        };
        for(int i: idx)
            label[i] = (h-1)*gs*gs + bin(ey,py[i])*gs + bin(ex,px[i]);
    }

    template<typename T> static void put(std::vector<uint8_t>&b,const T&v) {
        put_raw(b,&v,sizeof(T));
    }
    template<typename T> static void put_vec(std::vector<uint8_t>&b,const std::vector<T>&v) {
        put_raw(b,v.data(),v.size()*sizeof(T));
    }
    static void put_raw(std::vector<uint8_t>&b,const void*p,size_t n) {
        const uint8_t*c = (const uint8_t*)p;
        b.insert(b.end(),c,c+n);
    }
    template<typename T> static void get(const std::vector<uint8_t>&b,size_t&pos,T&v) {
        get_raw(b,pos,&v,sizeof(T));
    }
    template<typename T> static void get_vec(const std::vector<uint8_t>&b,size_t&pos,std::vector<T>&v,size_t n) {
        v.resize(n);
        get_raw(b,pos,v.data(),n*sizeof(T));
    }
    static void get_raw(const std::vector<uint8_t>&b,size_t&pos,void*p,size_t n) {
        memcpy(p,b.data()+pos,n);
        pos += n;
    }
};

#endif /// STATS_GROUPS_H
