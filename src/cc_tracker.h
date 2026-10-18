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
// Orientation prior weight for the alignment trackers.
//
// Computes a Bayesian-style scalar weight w ∈ (0, 1] that down-weights
// candidate orientations whose out-of-plane (cone) deviation from the
// previous estimate is large.  Intended to be multiplied into the CC / PSR
// score by the caller (or passed as the prior_weight argument to the
// tracker's push()).
//
// Inputs (all in degrees):
//   theta_deg   — out-of-plane angle of the candidate; typically computed by
//                 the caller as acos(R(2,2)) on the rotation matrix that
//                 represents the candidate's deviation from identity, then
//                 converted to degrees.  Only the cone component is penalised
//                 — pure in-plane rotations of R leave R(2,2)=1 and so
//                 produce theta_deg=0.
//   sigma_deg   — width of the prior.  Larger σ ⇒ weaker prior.  σ ≤ 0
//                 disables the regulariser (returns 1).  A reasonable range
//                 is σ ≈ cone_range/4 (aggressive: edge of the cone gets
//                 w ≈ 0.14) to σ ≈ cone_range/2 (mild: edge gets w ≈ 0.61).
//
// Default form is a Gaussian, w(θ) = exp(−θ² / 2σ²), corresponding to a
// Gaussian prior on the cone direction and yielding the MAP estimator when
// combined with a log-CC likelihood.  Alternatives the caller may swap in
// (left commented for reference):
//   • Lorentzian:   w(θ) = 1 / (1 + (θ/σ)²)
//                   — heavier tails, more permissive of genuine large
//                     offsets when the CC clearly supports them.
//   • Hard cutoff:  w(θ) = (θ ≤ σ) ? 1 : 0
//                   — strict windowed search, no soft prior.
//   • Inverse:      w(θ) = σ / (σ + θ)
//                   — milder linear falloff.
// ---------------------------------------------------------------------------
inline float orientation_prior_weight_deg(float theta_deg, float sigma_deg)
{
    if (sigma_deg <= 0.f) return 1.f;          // disabled
    const float r = theta_deg / sigma_deg;
    return std::exp(-0.5f * r * r);            // Gaussian
    // Lorentzian alternative:
    // return 1.f / (1.f + r * r);
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

// Pose: per orientation the offset with the highest weighted CC (offset prior on positive values),
// across orientations the highest weighted CC times the angle prior. get_cc(): the raw CC at that
// pose or, with CC_STATS_SIGMA, its z against the per-orientation best CC of the first level.
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
    };

    const Vec3* pts_;
    int    n_pts_;
    bool   sigma_stats_;
    bool   frozen_;
    int    k_;
    std::vector<float> pts_w_;
    std::vector<Entry> e_;

public:
    CcTrackerAlignmentArr(CcStatsType_t cc_stats,const int max_k,const Vec3*p_pts,const int n_pts,float offset_sigma=0.f)
        : pts_(p_pts), n_pts_(n_pts), sigma_stats_(cc_stats == CC_STATS_SIGMA), frozen_(false), k_(max_k)
    {
        pts_w_.assign(n_pts_,1.0f);
        if( offset_sigma > 0.f ) {
            const float inv2s2 = 1.0f/(2.0f*offset_sigma*offset_sigma);
            for(int i=0;i<n_pts_;i++) {
                const float r2 = pts_[i].x*pts_[i].x + pts_[i].y*pts_[i].y + pts_[i].z*pts_[i].z;
                pts_w_[i] = std::exp(-r2*inv2s2);
            }
        }
        e_.resize(max_k);
        clear(max_k);
    }

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

    void next_level() {
        frozen_ = true;
    }

    void push(const float*p_cc,const int n_pts,const M33f&Rot,float prior_weight=1.0f) {
        const int n = std::min(n_pts,n_pts_);
        if( n < 1 )
            return;
        for(int k=0;k<k_;k++) {
            const float*c = p_cc + k*n_pts;
            Entry&e = e_[k];
            float max_w = (c[0] > 0.f) ? c[0]*pts_w_[0] : c[0];
            int   max_i = 0;
            for(int i=1;i<n;i++) {
                const float v_w = (c[i] > 0.f) ? c[i]*pts_w_[i] : c[i];
                if( v_w > max_w ) { max_w = v_w; max_i = i; }
            }
            const float vmax = c[max_i];
            if( sigma_stats_ && !frozen_ ) {
                e.count++;
                const double d = vmax - e.mean;
                e.mean += d/e.count;
                e.m2   += d*(vmax - e.mean);
            }
            const float joint = (vmax > 0.f) ? max_w*prior_weight : vmax;
            if( joint > e.cc_w ) {
                e.cc_w = joint;
                e.cc   = vmax;
                e.vec  = pts_[max_i];
                e.rot  = Rot;
            }
        }
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
};

class CcTrackerAlignment {
private:
    CcTrackerAlignmentArr arr_;

public:
    CcTrackerAlignment(CcStatsType_t cc_stats,const Vec3*p_pts,const int n_pts,float offset_sigma=0.f)
        : arr_(cc_stats,1,p_pts,n_pts,offset_sigma) {}

    void clear()      { arr_.clear(1); }
    void next_level() { arr_.next_level(); }
    void push(const float*p_cc,const int n_pts,const M33f&Rot,float prior_weight=1.0f) { arr_.push(p_cc,n_pts,Rot,prior_weight); }
    float get_cc()  const { return arr_.get_cc(0);  }
    Vec3  get_vec() const { return arr_.get_vec(0); }
    M33f  get_rot() const { return arr_.get_rot(0); }
};


// ---------------------------------------------------------------------------
class CcTrackDefocusMax {
private:
    const Vec3* pts_;
    int   n_pts_;
    float offset_sigma_;
    float defocus_sigma_;
    float phase_sigma_;
    std::vector<float> pts_w_;     // precomputed shift-prior weights

    // Per-tilt best
    float        best_cc_;           // raw CC at chosen point (returned by get_cc)
    float        best_cc_weighted_;  // joint score — argmax tiebreak only
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
          offset_sigma_(offset_sigma),
          defocus_sigma_(defocus_sigma),
          phase_sigma_(phase_sigma)
    {
        pts_w_.assign(n_pts_, 1.0f);
        if (offset_sigma_ > 0.f) {
            const float inv2s2 = 1.0f / (2.0f * offset_sigma_ * offset_sigma_);
            for (int i = 0; i < n_pts_; ++i) {
                const float r2 = pts_[i].x*pts_[i].x +
                                 pts_[i].y*pts_[i].y +
                                 pts_[i].z*pts_[i].z;
                pts_w_[i] = std::exp(-r2 * inv2s2);
            }
        }
        clear();
    }

    void clear()
    {
        best_cc_          = -std::numeric_limits<float>::infinity();
        best_cc_weighted_ = -std::numeric_limits<float>::infinity();
        best_vec_         = {0.f, 0.f, 0.f};
        best_def_         = {0.f, 0.f, 0.f, 0.f};
    }

    void push(const float* p_cc,
              int n_pts,
              const DefocusDelta& delta)
    {
        const int n = std::min(n_pts, n_pts_);

        // Per-translation weighted argmax (joint with shift prior; sign-safe).
        const float v0_w = (p_cc[0] > 0.f) ? p_cc[0] * pts_w_[0] : p_cc[0];
        float vmax_w     = v0_w;
        int   max_idx    = 0;
        for (int i = 1; i < n; ++i) {
            const float v_w = (p_cc[i] > 0.f) ? p_cc[i] * pts_w_[i] : p_cc[i];
            if (v_w > vmax_w) { vmax_w = v_w; max_idx = i; }
        }
        const float vmax = p_cc[max_idx];   // raw CC at chosen translation

        // Defocus and phase priors (this push).
        float w_def = 1.f, w_phs = 1.f;
        if (defocus_sigma_ > 0.f) {
            const float r2 = delta.U*delta.U + delta.V*delta.V;
            w_def = std::exp(-0.5f * r2 / (defocus_sigma_ * defocus_sigma_));
        }
        if (phase_sigma_ > 0.f) {
            const float dP = delta.phase_shift_rad;
            w_phs = std::exp(-0.5f * dP*dP / (phase_sigma_ * phase_sigma_));
        }

        // Joint score for the cross-push tiebreak.  Sign-safe: when the raw
        // CC at the chosen point is non-positive, skip all multiplicative
        // weighting (matches the alignment tracker convention).
        const float vmax_joint = (vmax > 0.f) ? vmax_w * w_def * w_phs : vmax;

        if (vmax_joint > best_cc_weighted_) {
            best_cc_weighted_ = vmax_joint;
            best_cc_          = vmax;
            best_vec_         = pts_[max_idx];
            best_def_         = delta;
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

