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

#ifndef PEAK_STATS_H
#define PEAK_STATS_H

/// Per projection statistics of the 2D CC evidence (aligner type 2, -output_type alignment_stats).
/// C (P): raw CC (not weighted by prj_w, no priors) at the input orientation at every offset point (pixels).
/// Fields (NaN when undefined): cc, px, py, z_peak, h_xx, h_xy, h_yy, dx2, dy2, r2

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

namespace PeakStats {

const int NF = 10;
static const char* FIELDS[NF] = {"cc","px","py","z_peak","h_xx","h_xy","h_yy","dx2","dy2","r2"};

/// Reusable buffers of compute() (no allocations after the first calls).
struct Workspace {
    std::vector<int>    gx, gy;
    std::vector<double> V, us, vs;
};

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,0,9,6> MatQ;
typedef Eigen::Matrix<double,Eigen::Dynamic,1,0,9,1>              VecQ;
typedef Eigen::Matrix<double,6,1>                                 VecP;
typedef Eigen::Matrix<double,2,2>                                 MatH;
typedef Eigen::Matrix<double,2,1>                                 VecH;

/// C (P), pts on a grid of spacing `step` -> out (NF)
inline void compute(Workspace&ws,float*out,const float*C,int P,const float*pts_x,const float*pts_y,float step,float excl) {
    for(int i=0;i<NF;i++) out[i] = NAN;
    if( P == 0 ) return;

    ws.gx.resize(P);
    ws.gy.resize(P);
    int min_x=INT32_MAX, min_y=INT32_MAX, max_x=INT32_MIN, max_y=INT32_MIN;
    for(int p=0;p<P;p++) {
        ws.gx[p] = (int)lrintf(pts_x[p]/step);
        ws.gy[p] = (int)lrintf(pts_y[p]/step);
        min_x = std::min(min_x,ws.gx[p]); max_x = std::max(max_x,ws.gx[p]);
        min_y = std::min(min_y,ws.gy[p]); max_y = std::max(max_y,ws.gy[p]);
    }
    int nx = max_x-min_x+1, ny = max_y-min_y+1;
    ws.V.assign((size_t)nx*ny,-INFINITY);
    for(int p=0;p<P;p++)
        ws.V[(size_t)(ws.gy[p]-min_y)*nx + (ws.gx[p]-min_x)] = C[p];
    auto Vat = [&](int iy,int ix)->double {
        if( iy<0 || iy>=ny || ix<0 || ix>=nx ) return -INFINITY;
        return ws.V[(size_t)iy*nx+ix];
    };

    int p0=0;
    double c0=-INFINITY;
    for(int p=0;p<P;p++)
        if( std::isfinite(C[p]) && C[p] > c0 ) { c0 = C[p]; p0=p; }
    if( !std::isfinite(c0) ) return;
    out[0] = c0;
    int ix0 = ws.gx[p0]-min_x, iy0 = ws.gy[p0]-min_y;

    /// quadratic least squares on the 3x3 neighbourhood: v = c + g.u + 1/2 u^T H u
    ws.us.clear();
    ws.vs.clear();
    for(int dy=-1;dy<=1;dy++)
        for(int dx=-1;dx<=1;dx++) {
            double v = Vat(iy0+dy,ix0+dx);
            if( std::isfinite(v) ) {
                ws.us.push_back(dx*step); ws.us.push_back(dy*step);
                ws.vs.push_back(v);
            }
        }
    double px = pts_x[p0], py = pts_y[p0];
    int ns = (int)ws.vs.size();
    if( ns >= 6 ) {
        MatQ Am(ns,6);
        VecQ bv(ns);
        for(int s=0;s<ns;s++) {
            double x=ws.us[2*s], y=ws.us[2*s+1];
            Am(s,0)=1; Am(s,1)=x; Am(s,2)=y; Am(s,3)=0.5*x*x; Am(s,4)=x*y; Am(s,5)=0.5*y*y;
            bv(s) = ws.vs[s];
        }
        Eigen::ColPivHouseholderQR<MatQ> qr(Am);
        if( qr.rank() == 6 ) {   /// else degenerate neighbourhood
            VecP c = qr.solve(bv);
            MatH H;
            VecH g;
            H(0,0)=c(3); H(0,1)=H(1,0)=c(4); H(1,1)=c(5);
            g(0)=c(1); g(1)=c(2);
            out[4]=H(0,0); out[5]=H(0,1); out[6]=H(1,1);
            Eigen::SelfAdjointEigenSolver<MatH> es(H);
            if( es.eigenvalues().maxCoeff() < 0 ) {
                VecH du = -H.ldlt().solve(g);
                if( std::fabs(du(0))<=step && std::fabs(du(1))<=step ) { px += du(0); py += du(1); }
            }
        }
    }
    out[1]=px; out[2]=py;

    /// prominence over the offsets
    double row_mean = 0;
    {
        double s=0, s2=0; int cnt=0;
        for(int p=0;p<P;p++) { double v=C[p]; if(std::isfinite(v)) { s+=v; s2+=v*v; cnt++; } }
        if( cnt > 0 ) row_mean = s/cnt;
        if( cnt > 2 ) {
            double m=s/cnt, sd=std::sqrt(std::max(s2/cnt-m*m,0.0));
            if( sd > 0 ) out[3] = (c0-m)/sd;
        }
    }
    /// second peak exclusion radius: excl > 0 as given, else 3 sigma of the main peak
    /// (sigma^2 = peak height above the mean / sqrt(det(-H))), at least one grid step (2 steps without a Hessian)
    if( !(excl > 0) ) {
        excl = 2*step;
        double dH = (double)out[4]*out[6] - (double)out[5]*out[5];
        if( std::isfinite(out[4]) && out[4] < 0 && dH > 0 )
            excl = std::max((double)step, 3.0*std::sqrt(std::max(c0-row_mean,1e-12)/std::sqrt(dH)));
    }

    /// second peak: the highest local maximum (3x3) farther than excl px from the main peak
    double best=-INFINITY; int by=0,bx=0;
    for(int iy=0;iy<ny;iy++)
        for(int ix=0;ix<nx;ix++) {
            double v = Vat(iy,ix);
            if( !std::isfinite(v) ) continue;
            if( std::hypot((ix-ix0)*step,(iy-iy0)*step) <= excl ) continue;
            bool is_max = true;
            for(int dy=-1;dy<=1 && is_max;dy++)
                for(int dx=-1;dx<=1;dx++)
                    if( Vat(iy+dy,ix+dx) > v ) { is_max=false; break; }
            if( is_max && v > best ) { best=v; by=iy; bx=ix; }
        }
    if( std::isfinite(best) ) {
        out[7] = (bx-ix0)*step;
        out[8] = (by-iy0)*step;
        out[9] = (c0 != 0) ? best/c0 : NAN;
    }
}

}

#endif /// PEAK_STATS_H
