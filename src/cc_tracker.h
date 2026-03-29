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

class CcTrackAlignmentMax {
private:
    const Vec3* pts_;
    int   n_pts_;
    int   n_ang_max_;
    float pix_size_;
    float step_;
    bool  is_2d_;

    float current_cc_;
    float current_sigma_;
    Vec3  current_vec_;
    M33f  current_rot_;

public:
    CcTrackAlignmentMax(const Vec3* p_pts,
                        int n_pts,
                        int n_ang_max,
                        float pix_size)
        : pts_(p_pts),
        n_pts_(n_pts),
        n_ang_max_(n_ang_max),
        pix_size_(pix_size),
        step_(cc_tracker_detail::compute_step(p_pts, n_pts)),
        is_2d_(true)
    {
        for (int i = 0; i < n_pts_; ++i)
            if (pts_[i].z != 0.f) { is_2d_ = false; break; }
        clear();
    }

    void clear()
    {
        current_cc_    = -std::numeric_limits<float>::infinity();
        current_sigma_ = 0.f;
        current_vec_   = {0.f, 0.f, 0.f};
        current_rot_   = M33f::Identity();
    }

    void push(const float* p_cc,
              int n_pts,
              const M33f& Rot)
    {
        const int n = std::min(n_pts, n_pts_);

        float vmax    = p_cc[0];
        int   max_idx = 0;
        for (int i = 1; i < n; ++i)
            if (p_cc[i] > vmax) { vmax = p_cc[i]; max_idx = i; }

        if (vmax > current_cc_) {
            current_cc_    = vmax;
            current_vec_   = pts_[max_idx];
            current_rot_   = Rot;
            current_sigma_ = cc_tracker_detail::peak_sigma(pts_, n, p_cc, step_, is_2d_, pts_[max_idx]);
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

    // Best pose based on PSR
    float best_psr_;
    float best_sigma_;
    Vec3  best_vec_;
    M33f  best_rot_;

    // Welford stats over PSR (across angles)
    int   psr_count_;
    float psr_mean_;
    float psr_M2_;

public:
    CcTrackAlignmentSigma(const Vec3* p_pts,
                          int n_pts,
                          int n_ang_max,
                          float pix_size)
        : pts_(p_pts),
        n_pts_(n_pts),
        n_ang_max_(n_ang_max),
        pix_size_(pix_size),
        step_(cc_tracker_detail::compute_step(p_pts, n_pts)),
        is_2d_(true)
    {
        for (int i = 0; i < n_pts_; ++i)
            if (pts_[i].z != 0.f) { is_2d_ = false; break; }
        clear();
    }

    void clear()
    {
        best_psr_   = -std::numeric_limits<float>::infinity();
        best_sigma_ = 0.f;
        best_vec_   = {0.f,0.f,0.f};
        best_rot_   = M33f::Identity();

        psr_count_ = 0;
        psr_mean_  = 0.f;
        psr_M2_    = 0.f;
    }

    void push(const float* p_cc,
              int n_pts,
              const M33f& Rot)
    {
        const int n = std::min(n_pts, n_pts_);

        // Need at least 3 points: with n=2, PSR is provably always 1.0
        // regardless of the CC values (max - mean = stddev algebraically),
        // which gives no useful information about peak quality.
        if (n <= 2)
            return;

        // ---- First level (per push) ----
        // Find the true max without clamping so max_idx is always the actual
        // best CC point (not an arbitrary clamped-to-zero neighbour).
        float max_val = p_cc[0];
        int   max_idx = 0;
        for (int i = 1; i < n; ++i)
            if (p_cc[i] > max_val) { max_val = p_cc[i]; max_idx = i; }

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

        // Track best pose
        if (psr > best_psr_) {
            best_psr_   = psr;
            best_vec_   = pts_[max_idx];
            best_rot_   = Rot;
            best_sigma_ = cc_tracker_detail::peak_sigma(pts_, n, p_cc, step_, is_2d_, pts_[max_idx]);
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
    static tracker_t make_tracker(CcStatsType_t type,const Vec3* p_pts,int n_pts,const int n_ang_max,const float pix_size) {
        switch(type) {
            case CC_STATS_NONE:
                return CcTrackAlignmentMax(p_pts,n_pts,n_ang_max,pix_size);
            case CC_STATS_SIGMA:
                return CcTrackAlignmentSigma(p_pts,n_pts,n_ang_max,pix_size);
            default:
                return CcTrackAlignmentMax(p_pts,n_pts,n_ang_max,pix_size);
        }
    }

public:
    explicit CcTrackerAlignment(CcStatsType_t cc_stat_type,const Vec3*p_pts,const int n_pts,const int n_ang_max,const float pix_size)
        : tracker(make_tracker(cc_stat_type,p_pts,n_pts,n_ang_max,pix_size))
    {
    }

    void clear() {
        std::visit([&](auto&t){t.clear();},tracker);
    }

    void push(const float*p_cc,const int n_pts,const M33f&Rot) {
        std::visit([&](auto&t){t.push(p_cc,n_pts,Rot);},tracker);
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
                          float pix_size)
        : numel_(k)
    {
        trackers_.reserve(numel_);
        for (int i = 0; i < numel_; ++i) {
            trackers_.emplace_back(type, p_pts, n_pts, n_ang_max,pix_size);
        }
    }

    void clear()
    {
        for (auto& t : trackers_)
            t.clear();
    }

    void push(const float* p_cc,
              int n_pts,
              const M33f& Rot)
    {
        for (int i = 0; i < numel_; ++i) {
            int off = i * n_pts;
            trackers_[i].push(p_cc + off, n_pts, Rot);
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

#endif /// CC_TRACKER_H

