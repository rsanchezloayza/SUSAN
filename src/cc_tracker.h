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

#ifndef CC_TRACKER_H
#define CC_TRACKER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <cmath>
#include <algorithm>
#include <vector>

#include "datatypes.h"
#include "math_cpu.h"

// ---------------------------------------------------------------------------
// Priors of the alignment trackers, in log-likelihood units.
//
// The score of a candidate pose is CC - lambda * (|t|^2 / 2 sigma_t^2 + theta_cone^2 / 2 sigma_a^2 +
// theta_inplane^2 / 2 sigma_a^2): a log posterior in CC units. lambda = sd_cc^2 / A converts the CC to a
// log-likelihood (kappa CC, kappa = A / sd_cc^2; A the peak above the baseline, sd_cc the CC noise), measured by the
// aligner per projection (2D) or per particle (3D) on a ring of offsets outside the peak (AliData noise points);
// lambda = 0 disables the prior. The former multiplicative weights (CC exp(-d^2 / 2 sigma^2)) were z^2 = (A / sd_cc)^2
// too strong: on EMPIAR-10064 (z 3-3.5) sigma 1 px acted as about 0.3 px. Angles in degrees, offsets in pixels.
// ---------------------------------------------------------------------------
inline float orientation_prior_pen_deg(float theta_deg, float sigma_deg)
{
    if (sigma_deg <= 0.f) return 0.f;          // disabled
    const float r = theta_deg / sigma_deg;
    return 0.5f * r * r;
}

/// lambda = sd^2 / max(A, sd): sd = 1.4826 MAD of the CC on the noise ring (n_noise values, modified: partially
/// sorted), A = the best CC over the search points (n_pts) minus the ring median. The max() keeps a projection without
/// signal at the prior centre.
inline float prior_lambda(float*noise,const int n_noise,const float*cc,const int n_pts)
{
    if( n_noise < 3 || n_pts < 1 ) return 0.f;
    const int h = n_noise/2;
    std::nth_element(noise,noise+h,noise+n_noise);
    const float med = noise[h];
    for(int i=0;i<n_noise;i++) noise[i] = std::fabs(noise[i]-med);
    std::nth_element(noise,noise+h,noise+n_noise);
    const float sd = 1.4826f*noise[h];
    if( !(sd > 0.f) ) return 0.f;
    float best = cc[0];
    for(int i=1;i<n_pts;i++) best = std::max(best,cc[i]);
    return sd*sd/std::max(best-med,sd);
}

// Convenience: extract the cone polar deviation (in degrees) from a rotation
// matrix that represents a candidate's deviation from the previous pose.
// Returns acos(clamp(R(2,2), −1, 1)) in degrees.  Pure in-plane rotations
// give 0 (R(2,2) = 1), so the prior only penalises out-of-plane tilts.
inline float cone_theta_deg(const M33f& R)
{
    float c = R(2,2);
    if (c >  1.f) c =  1.f;
    if (c < -1.f) c = -1.f;
    return std::acos(c) * (180.f / float(M_PI));
}

// Convenience: extract the in-plane (twist about z) deviation, in degrees,
// from a rotation matrix that represents a candidate's deviation from the
// previous pose.  This is the twist component of the swing-twist
// decomposition about z: atan2(R(1,0)−R(0,1), R(0,0)+R(1,1)).  Pure
// out-of-plane tilts (Ry) give 0, so it is the complement of cone_theta_deg
// — the two together cover the full orientation deviation.
inline float inplane_theta_deg(const M33f& R)
{
    const float a = std::atan2(R(1,0) - R(0,1), R(0,0) + R(1,1));
    return std::fabs(a) * (180.f / float(M_PI));
}

// Pose: the highest score CC - lambda (|t|^2 / 2 sigma_t^2 + angle penalty) over the orientations and offsets
// (see prior_lambda; lambda per entry, 0 = no prior). get_cc(): the raw CC at that pose or, with CC_STATS_SIGMA, its
// z against the per-orientation best CC of the first level.
class CcTrackerAlignmentArr {
private:
    struct Entry {
        float  cc;
        float  cc_w;
        Vec3   vec;
        M33f   rot;
        int    count;
        double mean;
        double m2;
        float  lambda;
    };

    const Vec3* pts_;
    int    n_pts_;
    bool   sigma_stats_;
    bool   frozen_;
    int    k_;
    std::vector<float> pts_pen_;
    std::vector<Entry> e_;

public:
    CcTrackerAlignmentArr(CcStatsType_t cc_stats,const int max_k,const Vec3*p_pts,const int n_pts,float offset_sigma=0.f)
        : pts_(p_pts), n_pts_(n_pts), sigma_stats_(cc_stats == CC_STATS_SIGMA), frozen_(false), k_(max_k)
    {
        pts_pen_.assign(n_pts_,0.0f);
        if( offset_sigma > 0.f ) {
            const float inv2s2 = 1.0f/(2.0f*offset_sigma*offset_sigma);
            for(int i=0;i<n_pts_;i++) {
                const float r2 = pts_[i].x*pts_[i].x + pts_[i].y*pts_[i].y + pts_[i].z*pts_[i].z;
                pts_pen_[i] = r2*inv2s2;
            }
        }
        e_.resize(max_k);
        for(auto&e : e_) e.lambda = 0.f;
        clear(max_k);
    }

    /// Resets the poses and statistics; the lambdas are kept (set them per particle with set_lambda).
    void clear(const int k) {
        k_ = std::min(k,(int)e_.size());
        frozen_ = false;
        for(int i=0;i<k_;i++) {
            Entry&e = e_[i];
            e.cc    = -std::numeric_limits<float>::infinity();
            e.cc_w  = -std::numeric_limits<float>::infinity();
            e.vec   = {0.f,0.f,0.f};
            e.rot   = M33f::Identity();
            e.count = 0;
            e.mean  = 0;
            e.m2    = 0;
        }
    }

    void set_lambda(const int idx,const float lambda) { e_[idx].lambda = lambda; }

    void next_level() {
        frozen_ = true;
    }

    void push(const float*p_cc,const int n_pts,const M33f&Rot,float prior_pen=0.0f) {
        const int n = std::min(n_pts,n_pts_);
        for(int k=0;k<k_;k++)
            push_entry(e_[k],p_cc + k*n_pts,n,Rot,prior_pen);
    }

    void push(const float*p_cc,const int n_pts,const M33f*Rots,const float*prior_pens) {
        const int n = std::min(n_pts,n_pts_);
        for(int k=0;k<k_;k++)
            push_entry(e_[k],p_cc + k*n_pts,n,Rots[k],prior_pens[k]);
    }

    float get_cc(const int idx) const {
        const Entry&e = e_[idx];
        if( e.cc == -std::numeric_limits<float>::infinity() )
            return 0.f;
        if( !sigma_stats_ )
            return e.cc;
        if( e.count < 2 )
            return 0.f;
        const double var = e.m2/(e.count-1);
        if( var <= 0 )
            return 0.f;
        return (float)std::max((e.cc-e.mean)/std::sqrt(var),0.0);
    }

    Vec3 get_vec(const int idx) const { return e_[idx].vec; }
    M33f get_rot(const int idx) const { return e_[idx].rot; }
    int  size() const { return k_; }

private:
    void push_entry(Entry&e,const float*c,const int n,const M33f&Rot,const float prior_pen) {
        if( n < 1 )
            return;
        const float lam = e.lambda;
        float max_w = c[0] - lam*(pts_pen_[0] + prior_pen);
        int   max_i = 0;
        for(int i=1;i<n;i++) {
            const float v_w = c[i] - lam*(pts_pen_[i] + prior_pen);
            if( v_w > max_w ) { max_w = v_w; max_i = i; }
        }
        const float vmax = c[max_i];
        if( sigma_stats_ && !frozen_ ) {
            e.count++;
            const double d = vmax - e.mean;
            e.mean += d/e.count;
            e.m2   += d*(vmax - e.mean);
        }
        if( max_w > e.cc_w ) {
            e.cc_w = max_w;
            e.cc   = vmax;
            e.vec  = pts_[max_i];
            e.rot  = Rot;
        }
    }
};

class CcTrackerAlignment {
private:
    CcTrackerAlignmentArr arr_;

public:
    CcTrackerAlignment(CcStatsType_t cc_stats,const Vec3*p_pts,const int n_pts,float offset_sigma=0.f)
        : arr_(cc_stats,1,p_pts,n_pts,offset_sigma) {}

    void clear()      { arr_.clear(1); }
    void next_level() { arr_.next_level(); }
    void set_lambda(const float lambda) { arr_.set_lambda(0,lambda); }
    void push(const float*p_cc,const int n_pts,const M33f&Rot,float prior_pen=0.0f) { arr_.push(p_cc,n_pts,Rot,prior_pen); }
    float get_cc()  const { return arr_.get_cc(0);  }
    Vec3  get_vec() const { return arr_.get_vec(0); }
    M33f  get_rot() const { return arr_.get_rot(0); }
};


// ---------------------------------------------------------------------------
// Per-projection best (offset, defocus delta) of the CTF refiner. Score: CC - lambda (|t|^2 / 2 sigma_t^2 +
// (dU^2 + dV^2) / 2 sigma_def^2 + dP^2 / 2 sigma_ph^2), lambda = sd_cc^2 / A of the projection (prior_lambda; 0 =
// no prior), as the alignment trackers. get_cc() is the raw CC at the chosen point.
class CcTrackDefocusMax {
private:
    const Vec3* pts_;
    int   n_pts_;
    float defocus_sigma_;
    float phase_sigma_;
    float lambda_;
    std::vector<float> pts_pen_;     // precomputed shift-prior penalties

    float        best_cc_;           // raw CC at chosen point (returned by get_cc)
    float        best_score_;
    Vec3         best_vec_;
    DefocusDelta best_def_;

public:
    CcTrackDefocusMax(const Vec3* p_pts,
                      int n_pts,
                      float offset_sigma = 0.f,
                      float defocus_sigma = 0.f,
                      float phase_sigma   = 0.f)
        : pts_(p_pts),
          n_pts_(n_pts),
          defocus_sigma_(defocus_sigma),
          phase_sigma_(phase_sigma),
          lambda_(0.f)
    {
        pts_pen_.assign(n_pts_, 0.0f);
        if (offset_sigma > 0.f) {
            const float inv2s2 = 1.0f / (2.0f * offset_sigma * offset_sigma);
            for (int i = 0; i < n_pts_; ++i) {
                const float r2 = pts_[i].x*pts_[i].x +
                                 pts_[i].y*pts_[i].y +
                                 pts_[i].z*pts_[i].z;
                pts_pen_[i] = r2 * inv2s2;
            }
        }
        clear();
    }

    /// Resets the best point; lambda is kept.
    void clear()
    {
        best_cc_    = -std::numeric_limits<float>::infinity();
        best_score_ = -std::numeric_limits<float>::infinity();
        best_vec_   = {0.f, 0.f, 0.f};
        best_def_   = {0.f, 0.f, 0.f, 0.f};
    }

    void set_lambda(const float lambda) { lambda_ = lambda; }

    void push(const float* p_cc,
              int n_pts,
              const DefocusDelta& delta)
    {
        const int n = std::min(n_pts, n_pts_);
        if (n < 1) return;

        float pen = 0.f;
        if (defocus_sigma_ > 0.f)
            pen += (delta.U*delta.U + delta.V*delta.V) / (2.0f * defocus_sigma_ * defocus_sigma_);
        if (phase_sigma_ > 0.f)
            pen += delta.phase_shift_rad*delta.phase_shift_rad / (2.0f * phase_sigma_ * phase_sigma_);

        float smax    = p_cc[0] - lambda_ * (pts_pen_[0] + pen);
        int   max_idx = 0;
        for (int i = 1; i < n; ++i) {
            const float sc = p_cc[i] - lambda_ * (pts_pen_[i] + pen);
            if (sc > smax) { smax = sc; max_idx = i; }
        }

        if (smax > best_score_) {
            best_score_ = smax;
            best_cc_    = p_cc[max_idx];
            best_vec_   = pts_[max_idx];
            best_def_   = delta;
        }
    }

    float        get_cc()  const { return (best_cc_ == -std::numeric_limits<float>::infinity()) ? 0.f : best_cc_; }
    Vec3         get_vec() const { return best_vec_; }
    DefocusDelta get_def() const { return best_def_; }
};


class CcTrackerDefocusArrMax {
private:
    std::vector<CcTrackDefocusMax> trackers_;
    int numel_;

public:
    CcTrackerDefocusArrMax(int k,
                           const Vec3* p_pts,
                           int n_pts,
                           float offset_sigma = 0.f,
                           float defocus_sigma = 0.f,
                           float phase_sigma   = 0.f)
        : numel_(k)
    {
        trackers_.reserve(numel_);
        for (int i = 0; i < numel_; ++i) {
            trackers_.emplace_back(p_pts, n_pts, offset_sigma, defocus_sigma, phase_sigma);
        }
    }

    void clear()
    {
        for (auto& t : trackers_) t.clear();
    }

    void set_lambda(int i, const float lambda) { trackers_[i].set_lambda(lambda); }

    void push(const float* p_cc,
              int n_pts,
              const DefocusDelta& delta)
    {
        for (int i = 0; i < numel_; ++i) {
            int off = i * n_pts;
            trackers_[i].push(p_cc + off, n_pts, delta);
        }
    }

    float        get_cc(int i)  const { return trackers_[i].get_cc();  }
    Vec3         get_vec(int i) const { return trackers_[i].get_vec(); }
    DefocusDelta get_def(int i) const { return trackers_[i].get_def(); }

    int size() const { return numel_; }
};


#endif /// CC_TRACKER_H

