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
#include <variant>
#include <vector>

#include "datatypes.h"
#include "math_cpu.h"

// ---------------------------------------------------------------------------
// Helper: estimate the RMS half-width σ of the CC peak via local curvature.
//
// For a Gaussian f(x)=A·exp(−x²/2σ²)+c, the second finite difference of
// log(f−c) along any axis satisfies:
//
//   log(v₊) + log(v₋) − 2·log(v_c) = −step²/σ²
//
// where v_c = CC(peak)−c, v₊ = CC(peak+step·ê)−c, etc.  The subpixel
// offset x₀ cancels exactly, so the estimate is independent of the peak
// position within the grid cell, the amplitude A, and the window radius.
//
// Only the immediate cardinal neighbours (±step along each axis) are used,
// so the result is completely independent of the integration window.
// ---------------------------------------------------------------------------
namespace cc_tracker_detail {

inline float compute_step(const Vec3* pts, int n)
{
    float min_d2 = std::numeric_limits<float>::max();
    for (int i = 1; i < n; ++i) {
        float dx = pts[i].x - pts[0].x;
        float dy = pts[i].y - pts[0].y;
        float dz = pts[i].z - pts[0].z;
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > 0.f && d2 < min_d2) min_d2 = d2;
    }
    return (min_d2 < std::numeric_limits<float>::max()) ? std::sqrt(min_d2) : 1.f;
}

// peak_pos: the known discrete-maximum location (passed in by the caller,
// which already found max_idx, so no need to search again).
inline float peak_sigma(const Vec3* pts, int n, const float* p_cc,
                        float step, bool is_2d, const Vec3& peak_pos)
{
    const float tol   = 0.1f * step;    // grid-point matching tolerance
    const float step2 = step * step;

    // ---- 1. Peak CC value ----
    float v_c = 0.f;
    for (int i = 0; i < n; ++i) {
        float dx = pts[i].x - peak_pos.x;
        float dy = pts[i].y - peak_pos.y;
        float dz = pts[i].z - peak_pos.z;
        if (std::abs(dx) < tol && std::abs(dy) < tol && std::abs(dz) < tol)
            { v_c = p_cc[i]; break; }
    }

    // ---- 2. Cardinal neighbours: ±step along x, y, (z) ----
    // Store the found CC values for each of the 6 (or 4) directions.
    float vp[3] = {0.f, 0.f, 0.f};   // +x, +y, +z
    float vm[3] = {0.f, 0.f, 0.f};   // −x, −y, −z
    bool  fp[3] = {false,false,false};
    bool  fm[3] = {false,false,false};
    const int n_ax = is_2d ? 2 : 3;

    for (int i = 0; i < n; ++i) {
        float dx = pts[i].x - peak_pos.x;
        float dy = pts[i].y - peak_pos.y;
        float dz = pts[i].z - peak_pos.z;

        // x-axis neighbour: dy≈0, dz≈0, |dx|≈step
        if (std::abs(dy) < tol && std::abs(dz) < tol) {
            if (std::abs(dx - step) < tol) { vp[0] = p_cc[i]; fp[0] = true; }
            if (std::abs(dx + step) < tol) { vm[0] = p_cc[i]; fm[0] = true; }
        }
        // y-axis neighbour: dx≈0, dz≈0, |dy|≈step
        if (std::abs(dx) < tol && std::abs(dz) < tol) {
            if (std::abs(dy - step) < tol) { vp[1] = p_cc[i]; fp[1] = true; }
            if (std::abs(dy + step) < tol) { vm[1] = p_cc[i]; fm[1] = true; }
        }
        // z-axis neighbour (3D only): dx≈0, dy≈0, |dz|≈step
        if (!is_2d && std::abs(dx) < tol && std::abs(dy) < tol) {
            if (std::abs(dz - step) < tol) { vp[2] = p_cc[i]; fp[2] = true; }
            if (std::abs(dz + step) < tol) { vm[2] = p_cc[i]; fm[2] = true; }
        }
    }

    // ---- 3. Background: min over {peak, all found cardinal neighbours} ----
    float c = v_c;
    for (int a = 0; a < n_ax; ++a) {
        if (fp[a] && vp[a] < c) c = vp[a];
        if (fm[a] && vm[a] < c) c = vm[a];
    }
    c -= 1e-7f;   // ensure v_c − c > 0 strictly

    // ---- 4. Per-axis curvature → σ² = −step² / (log v₊ + log v₋ − 2 log v_c) ----
    const float log_vc = std::log(v_c - c);
    float sigma_sq_sum = 0.f;
    int   n_valid      = 0;

    for (int a = 0; a < n_ax; ++a) {
        if (!fp[a] || !fm[a]) continue;
        float lp = vp[a] - c, lm = vm[a] - c;
        if (lp <= 0.f || lm <= 0.f) continue;
        float curv = std::log(lp) + std::log(lm) - 2.f * log_vc;
        if (curv >= 0.f) continue;       // not concave → skip this axis
        sigma_sq_sum += -step2 / curv;
        ++n_valid;
    }

    if (n_valid == 0) return 0.f;
    float sigma_sq = sigma_sq_sum / float(n_valid);
    return (sigma_sq > 0.f) ? std::sqrt(sigma_sq) : 0.f;
}

} // namespace cc_tracker_detail

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

class CcTrackAlignmentMax {
private:
    const Vec3* pts_;
    int   n_pts_;
    int   n_ang_max_;
    float pix_size_;
    float step_;
    bool  is_2d_;
    float offset_sigma_;
    std::vector<float> pts_w_;  // precomputed translational prior weights

    float current_cc_;          // raw CC of the stored best (returned by get_cc)
    float current_cc_weighted_; // joint (CC · w_shift · w_angle) — argmax tiebreak only
    float current_sigma_;
    Vec3  current_vec_;
    M33f  current_rot_;

public:
    CcTrackAlignmentMax(const Vec3* p_pts,
                        int n_pts,
                        int n_ang_max,
                        float pix_size,
                        float offset_sigma = 0.f)
        : pts_(p_pts),
        n_pts_(n_pts),
        n_ang_max_(n_ang_max),
        pix_size_(pix_size),
        step_(cc_tracker_detail::compute_step(p_pts, n_pts)),
        is_2d_(true),
        offset_sigma_(offset_sigma)
    {
        for (int i = 0; i < n_pts_; ++i)
            if (pts_[i].z != 0.f) { is_2d_ = false; break; }
        // Precompute the translational prior weights w(t) = exp(−|t|²/(2σ²)).
        // pts_ are in pixels/voxels; |t|² degenerates to x²+y² in 2D since
        // pts_[i].z == 0 for all i.  σ ≤ 0 disables (weights set to 1).
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
        current_cc_          = -std::numeric_limits<float>::infinity();
        current_cc_weighted_ = -std::numeric_limits<float>::infinity();
        current_sigma_       = 0.f;
        current_vec_         = {0.f, 0.f, 0.f};
        current_rot_         = M33f::Identity();
    }

    // prior_weight: orientation prior (angle_sigma).  See angle_sigma docstring.
    // Expected range (0, 1].  prior_weight = 1.0 ⇒ orientation prior disabled.
    //
    // The translational prior (offset_sigma) is precomputed in pts_w_ and is
    // applied internally to the per-translation argmax:
    //     score_t(i) = p_cc[i] · pts_w_[i]              if p_cc[i] > 0
    //                = p_cc[i]                          if p_cc[i] ≤ 0   (sign-safe)
    // The cross-push tiebreak uses the joint MAP score:
    //     score(R) = score_t(t*) · prior_weight
    // i.e. both priors multiply.  get_cc() / get_vec() / get_rot() return the
    // raw values at the chosen (regularised) point, so neither prior pollutes
    // downstream stats.
    void push(const float* p_cc,
              int n_pts,
              const M33f& Rot,
              float prior_weight = 1.0f)
    {
        const int n = std::min(n_pts, n_pts_);

        // Per-translation weighted argmax (joint with shift prior).
        const float v0_w = (p_cc[0] > 0.f) ? p_cc[0] * pts_w_[0] : p_cc[0];
        float vmax_w     = v0_w;
        int   max_idx    = 0;
        for (int i = 1; i < n; ++i) {
            const float v_w = (p_cc[i] > 0.f) ? p_cc[i] * pts_w_[i] : p_cc[i];
            if (v_w > vmax_w) { vmax_w = v_w; max_idx = i; }
        }

        const float vmax       = p_cc[max_idx];                            // raw CC at chosen t
        const float vmax_joint = (vmax > 0.f) ? vmax_w * prior_weight      // full joint score
                                              : vmax;                       // sign-safe: skip both priors

        if (vmax_joint > current_cc_weighted_) {
            current_cc_weighted_ = vmax_joint;
            current_cc_          = vmax;
            current_vec_         = pts_[max_idx];
            current_rot_         = Rot;
            current_sigma_       = cc_tracker_detail::peak_sigma(pts_, n, p_cc, step_, is_2d_, pts_[max_idx]);
        }
    }

    float get_cc() const
    {
        if (current_cc_ == -std::numeric_limits<float>::infinity())
            return 0.f;
        return current_cc_;
    }

    Vec3 get_vec() const { return current_vec_; }
    M33f get_rot() const { return current_rot_; }

    float get_dose() const
    {
        if (current_sigma_ <= 0.f) return 9999.f;
        const float s    = current_sigma_ * pix_size_;
        const float dose = float(M_PI*M_PI) * s * s;
        return dose < 9999.f ? dose : 9999.f;
    }
};

class CcTrackAlignmentSigma {
private:
    const Vec3* pts_;
    int   n_pts_;
    int   n_ang_max_;
    float pix_size_;
    float step_;
    bool  is_2d_;
    float offset_sigma_;
    std::vector<float> pts_w_;  // precomputed translational prior weights

    // Best pose based on PSR (raw value stored; weighted only for tiebreak)
    float best_psr_;
    float best_psr_weighted_;
    float best_sigma_;
    Vec3  best_vec_;
    M33f  best_rot_;

    // Welford stats over PSR (across angles).  Accumulates RAW psr so the
    // z-score returned by get_cc() reflects the angular discriminability of
    // the data, independent of any orientation prior.
    int   psr_count_;
    float psr_mean_;
    float psr_M2_;

public:
    CcTrackAlignmentSigma(const Vec3* p_pts,
                          int n_pts,
                          int n_ang_max,
                          float pix_size,
                          float offset_sigma = 0.f)
        : pts_(p_pts),
        n_pts_(n_pts),
        n_ang_max_(n_ang_max),
        pix_size_(pix_size),
        step_(cc_tracker_detail::compute_step(p_pts, n_pts)),
        is_2d_(true),
        offset_sigma_(offset_sigma)
    {
        for (int i = 0; i < n_pts_; ++i)
            if (pts_[i].z != 0.f) { is_2d_ = false; break; }
        // See CcTrackAlignmentMax ctor: same w(t) = exp(−|t|²/(2σ²)) precomputation.
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
        best_psr_          = -std::numeric_limits<float>::infinity();
        best_psr_weighted_ = -std::numeric_limits<float>::infinity();
        best_sigma_        = 0.f;
        best_vec_          = {0.f,0.f,0.f};
        best_rot_          = M33f::Identity();

        psr_count_ = 0;
        psr_mean_  = 0.f;
        psr_M2_    = 0.f;
    }

    // prior_weight: orientation prior (angle_sigma); range (0, 1].  The
    // translational prior (offset_sigma) is precomputed in pts_w_.
    //
    // Per-translation argmax uses the shift-weighted score (sign-safe).  The
    // PSR numerator is the RAW p_cc[max_idx] at the chosen point (P1), so the
    // stored best_vec_ and PSR describe the same point.  Welford stats are
    // accumulated on RAW p_cc and RAW psr — neither prior leaks into them, so
    // get_cc() still reports a clean angular-discriminability z-score.
    // Cross-push tiebreak is the joint score:
    //     PSR · pts_w_[max_idx] · prior_weight
    // (PSR ≥ 0, so no sign guard is needed.)
    void push(const float* p_cc,
              int n_pts,
              const M33f& Rot,
              float prior_weight = 1.0f)
    {
        const int n = std::min(n_pts, n_pts_);

        // Need at least 3 points: with n=2, PSR is provably always 1.0
        // regardless of the CC values (max - mean = stddev algebraically),
        // which gives no useful information about peak quality.
        if (n <= 2)
            return;

        // ---- First level (per push) ----
        // Per-translation weighted argmax (joint with the offset_sigma prior).
        // Negative CCs bypass the multiplicative weight (sign-safe).
        const float v0_w = (p_cc[0] > 0.f) ? p_cc[0] * pts_w_[0] : p_cc[0];
        float max_val_w  = v0_w;
        int   max_idx    = 0;
        for (int i = 1; i < n; ++i) {
            const float v_w = (p_cc[i] > 0.f) ? p_cc[i] * pts_w_[i] : p_cc[i];
            if (v_w > max_val_w) { max_val_w = v_w; max_idx = i; }
        }
        const float max_val = p_cc[max_idx];  // raw CC at the chosen point (P1)

        // Welford online mean/variance over raw CC values: numerically stable,
        // avoids catastrophic cancellation in (sqsum/n − mean²).
        // Uses biased (n) denominator so that PSR_raw = sqrt(n-1) exactly for
        // an ideal single peak above uniform background, making the sqrt(n-1)
        // normalisation below yield 1.0 on that ideal case.
        float wf_mean = 0.f;
        float wf_M2   = 0.f;
        int   count   = 0;

        for (int i = 0; i < n; ++i) {
            float v = p_cc[i];
            count++;
            float d = v - wf_mean;
            wf_mean += d / count;
            wf_M2   += d * (v - wf_mean);
        }

        // count == n >= 3 here; use biased variance (consistent with PSR derivation)
        float var = wf_M2 / count;
        if (var <= 0.f)
            return;

        float stddev = std::sqrt(var);

        // Normalise PSR by sqrt(n-1) to remove grid-size dependence.
        // For an ideal single peak above uniform background, raw PSR equals
        // sqrt(n-1) (with biased variance), so after normalisation the value
        // is 1.0 regardless of n.  This makes stored PSR values comparable
        // across grids of different sizes and keeps the count==1 return and
        // the z-score return of get_cc() on a consistent scale.
        float psr = (max_val - wf_mean) / (stddev * std::sqrt(float(n - 1)));

        // ---- Second level (Welford across angles) ----
        psr_count_++;

        float delta  = psr - psr_mean_;
        psr_mean_   += delta / psr_count_;
        float delta2 = psr - psr_mean_;
        psr_M2_     += delta * delta2;

        // Track best pose — joint score for the cross-push tiebreak.
        const float psr_w = psr * pts_w_[max_idx] * prior_weight;
        if (psr_w > best_psr_weighted_) {
            best_psr_weighted_ = psr_w;
            best_psr_          = psr;
            best_vec_          = pts_[max_idx];
            best_rot_          = Rot;
            best_sigma_        = cc_tracker_detail::peak_sigma(pts_, n, p_cc, step_, is_2d_, pts_[max_idx]);
        }
    }

    float get_cc() const
    {
        if (psr_count_ == 0)
            return 0.f;

        if (psr_count_ == 1)
            return best_psr_;   // no variance yet

        float variance = psr_M2_ / (psr_count_ - 1);

        if (variance <= 0.f)
            // All rotations produced identical PSR → no angular
            // discriminability → we cannot identify a preferred pose.
            return 0.f;

        float stddev = std::sqrt(variance);

        return std::max((best_psr_ - psr_mean_) / stddev, 0.f);
    }

    Vec3 get_vec() const { return best_vec_; }
    M33f get_rot() const { return best_rot_; }

    float get_dose() const
    {
        if (best_sigma_ <= 0.f) return 9999.f;
        const float s    = best_sigma_ * pix_size_;
        const float dose = float(M_PI*M_PI) * s * s;
        return dose < 9999.f ? dose : 9999.f;
    }
};


class CcTrackerAlignment {
protected:
    using tracker_t = std::variant<CcTrackAlignmentMax,CcTrackAlignmentSigma>;
    tracker_t tracker;
protected:
    static tracker_t make_tracker(CcStatsType_t type,const Vec3* p_pts,int n_pts,const int n_ang_max,const float pix_size,float offset_sigma) {
        switch(type) {
            case CC_STATS_NONE:
                return CcTrackAlignmentMax(p_pts,n_pts,n_ang_max,pix_size,offset_sigma);
            case CC_STATS_SIGMA:
                return CcTrackAlignmentSigma(p_pts,n_pts,n_ang_max,pix_size,offset_sigma);
            default:
                return CcTrackAlignmentMax(p_pts,n_pts,n_ang_max,pix_size,offset_sigma);
        }
    }

public:
    explicit CcTrackerAlignment(CcStatsType_t cc_stat_type,const Vec3*p_pts,const int n_pts,const int n_ang_max,const float pix_size,float offset_sigma = 0.f)
        : tracker(make_tracker(cc_stat_type,p_pts,n_pts,n_ang_max,pix_size,offset_sigma))
    {
    }

    void clear() {
        std::visit([&](auto&t){t.clear();},tracker);
    }

    void push(const float*p_cc,const int n_pts,const M33f&Rot,float prior_weight = 1.0f) {
        std::visit([&](auto&t){t.push(p_cc,n_pts,Rot,prior_weight);},tracker);
    }

    float get_cc() const {
        return std::visit([](auto const& t) {return t.get_cc();}, tracker);
    }

    Vec3 get_vec() const {
        return std::visit([](auto const& t) {return t.get_vec();}, tracker);
    }

    M33f get_rot() const {
        return std::visit([](auto const& t) {return t.get_rot();}, tracker);
    }

    float get_dose() const {
        return std::visit([](auto const& t) {return t.get_dose();}, tracker);
    }
};

class CcTrackerAlignmentArr {
private:
    std::vector<CcTrackerAlignment> trackers_;
    int numel_;

public:
    CcTrackerAlignmentArr(CcStatsType_t type,
                          int k,
                          const Vec3* p_pts,
                          int n_pts,
                          int n_ang_max,
                          float pix_size,
                          float offset_sigma = 0.f)
        : numel_(k)
    {
        trackers_.reserve(numel_);
        for (int i = 0; i < numel_; ++i) {
            trackers_.emplace_back(type, p_pts, n_pts, n_ang_max, pix_size, offset_sigma);
        }
    }

    void clear()
    {
        for (auto& t : trackers_)
            t.clear();
    }

    void push(const float* p_cc,
              int n_pts,
              const M33f& Rot,
              float prior_weight = 1.0f)
    {
        for (int i = 0; i < numel_; ++i) {
            int off = i * n_pts;
            trackers_[i].push(p_cc + off, n_pts, Rot, prior_weight);
        }
    }

    float get_cc(int idx) const
    {
        return trackers_[idx].get_cc();
    }

    Vec3 get_vec(int idx) const
    {
        return trackers_[idx].get_vec();
    }

    M33f get_rot(int idx) const
    {
        return trackers_[idx].get_rot();
    }

    float get_dose(int idx) const
    {
        return trackers_[idx].get_dose();
    }

    int size() const { return numel_; }
};


// ---------------------------------------------------------------------------
// Defocus-search trackers.
//
// Used by ctf_refiner's nested (dU, dV, dA, dP) grid search.  Each push
// supplies a CC map for one (defocus, phase-shift) candidate, plus the
// candidate's DefocusDelta; the tracker keeps the joint-MAP-best per tilt.
//
// Three independent Gaussian priors are applied, all configured at
// construction (none change during the search):
//   • offset_sigma  (pixels)  — on translation magnitude |t|, precomputed
//                                 into pts_w_[i] = exp(−|t_i|² / 2σ²).
//   • defocus_sigma (Å)       — on isotropic defocus deviation, applied per
//                                 push as exp(−(dU² + dV²) / 2σ²).
//   • phase_sigma   (rad)     — on phase-shift magnitude, applied per push
//                                 as exp(−dP² / 2σ²).
// All sigmas ≤ 0 ⇒ that prior is disabled (weight = 1).
//
// Joint score for the cross-push tiebreak (Bayesian-clean):
//     score(R) = CC(t*) · w_shift(t*) · w_def(dU,dV) · w_phs(dP)
// Sign-safe with the same convention as the alignment trackers: when the
// raw CC at the chosen point is ≤ 0, weighting is skipped (otherwise the
// multiplicative weights would make a negative CC look "better").
// Returned values (get_cc / get_vec / get_def) are all raw at the chosen
// (regularised) point — no prior pollution downstream.
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

