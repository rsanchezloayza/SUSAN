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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>


#include <iostream>
#include <iomanip>

void printEigenMatrixAsPython(const Eigen::MatrixXf& M, const std::string& name)
{
    std::cout << name << " = np.array([\n";
    for (int i = 0; i < M.rows(); ++i)
    {
        std::cout << "    [";
        for (int j = 0; j < M.cols(); ++j)
        {
            std::cout << std::setprecision(8) << M(i,j);
            if (j < M.cols()-1) std::cout << ", ";
        }
        std::cout << "]";
        if (i < M.rows()-1) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "], dtype=np.float32)\n\n";
}

void printEigenVectorAsPython(const Eigen::VectorXf& v, const std::string& name)
{
    std::cout << name << " = np.array([";
    for (int i = 0; i < v.size(); ++i)
    {
        std::cout << std::setprecision(8) << v(i);
        if (i < v.size()-1) std::cout << ", ";
    }
    std::cout << "], dtype=np.float32)\n\n";
}

class CcTrackAlignmentMax {
private:
    const Vec3* pts_;      // external points (not owned)
    int n_pts_;            // number of points
    int n_ang_max_;        // not used here but stored if needed
    float pix_size_;

    float current_cc_;
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
        pix_size_(pix_size)
    {
        clear();
    }

    void clear()
    {
        current_cc_  = -std::numeric_limits<float>::infinity();
        current_vec_ = {0.f, 0.f, 0.f};
        current_rot_ = M33f::Identity();
    }

    void push(const float* p_cc,
              int n_pts,
              const M33f& Rot)
    {
        // Safety: ensure consistent point count
        const int n = std::min(n_pts, n_pts_);

        for (int i = 0; i < n; ++i) {

            float v = p_cc[i];

            if (v > current_cc_) {
                current_cc_  = v;
                current_vec_ = pts_[i];
                current_rot_ = Rot;
            }
        }
    }

    float get_cc() const
    {
        if (std::isinf(current_cc_))
            return 0.f;

        return current_cc_;
    }

    Vec3 get_vec() const
    {
        return current_vec_;
    }

    M33f get_rot() const
    {
        return current_rot_;
    }

    float get_dose() const
    {
        return 0.f;
    }
};

class CcTrackAlignmentSigma {
private:
    const Vec3* pts_;
    int n_pts_;
    int n_ang_max_;
    float pix_size_;

    // Best pose based on PSR
    float best_psr_;
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
        pix_size_(pix_size)
    {
        clear();
    }

    void clear()
    {
        best_psr_  = -std::numeric_limits<float>::infinity();
        best_vec_  = {0.f,0.f,0.f};
        best_rot_  = M33f::Identity();

        psr_count_ = 0;
        psr_mean_  = 0.f;
        psr_M2_    = 0.f;
    }

    void push(const float* p_cc,
              int n_pts,
              const M33f& Rot)
    {
        const int n = std::min(n_pts, n_pts_);

        if (n <= 1)
            return;

        // ---- First level (per push) ----
        float sum = 0.f;
        float sqsum = 0.f;
        float max_val = p_cc[0];
        int   max_idx = 0;

        for (int i = 0; i < n; ++i) {

            float v = std::max(p_cc[i], 0.f);

            sum += v;
            sqsum += v * v;

            if (v > max_val) {
                max_val = v;
                max_idx = i;
            }
        }

        float mean = sum / n;
        float var  = (sqsum / n) - mean * mean;

        if (var <= 0.f)
            return;

        float stddev = std::sqrt(var);
        if (stddev <= 0.f)
            return;

        float psr = (max_val - mean) / stddev;

        // ---- Second level (Welford across angles) ----
        psr_count_++;

        float delta  = psr - psr_mean_;
        psr_mean_   += delta / psr_count_;
        float delta2 = psr - psr_mean_;
        psr_M2_     += delta * delta2;

        // Track best pose
        if (psr > best_psr_) {
            best_psr_ = psr;
            best_vec_ = pts_[max_idx];
            best_rot_ = Rot;
        }
    }

    float get_cc() const
    {
        if (psr_count_ == 0)
            return 0.f;

        if (psr_count_ == 1)
            return best_psr_;   // no variance yet

        float variance = psr_M2_ / psr_count_;

        if (variance <= 0.f)
            return best_psr_;

        float stddev = std::sqrt(variance);

        return std::max((best_psr_ - psr_mean_) / stddev, 0.f);
    }

    Vec3 get_vec() const
    {
        return best_vec_;
    }

    M33f get_rot() const
    {
        return best_rot_;
    }

    float get_dose() const
    {
        return 0.f;
    }
};

class CcTrackAlignmentWgtAvg {
private:
    const Vec3* pts_;
    int n_pts_;
    float pix_size_;

    // global accumulators (across angles)
    float total_angle_weight_;
    Vec3  weighted_translation_sum_;
    Eigen::Matrix4f weighted_rotation_sum_;

    // PSR threshold (optional but recommended)
    float psr_threshold_;

public:
    CcTrackAlignmentWgtAvg(const Vec3* p_pts,
                                int n_pts,
                                int /*n_ang_max*/,
                                float pix_size,
                                float psr_threshold = 1e-5)
        : pts_(p_pts),
        n_pts_(n_pts),
        pix_size_(pix_size),
        psr_threshold_(psr_threshold)
    {
        clear();
    }

    void clear()
    {
        total_angle_weight_ = 0.f;
        weighted_translation_sum_ = {0.f,0.f,0.f};
        weighted_rotation_sum_    = Eigen::Matrix4f::Zero();
    }

    void push(const float* p_cc,
              int n_pts,
              const M33f& Rot)
    {
        const int n = std::min(n_pts, n_pts_);
        if (n <= 2)   // need at least 3 samples to exclude one
            return;

        // -----------------------------
        // 1. Find max
        // -----------------------------
        float max_val = p_cc[0];
        int   max_idx = 0;

        for (int i = 1; i < n; ++i) {
            if (p_cc[i] > max_val) {
                max_val = p_cc[i];
                max_idx = i;
            }
        }

        // -----------------------------
        // 2. Compute mean/std excluding max
        // -----------------------------
        float sum = 0.f;
        float sqsum = 0.f;
        int   count = 0;

        for (int i = 0; i < n; ++i) {
            if (i == max_idx)
                continue;

            float v = p_cc[i];
            sum   += v;
            sqsum += v * v;
            count++;
        }

        if (count <= 1)
            return;

        float mean = sum / count;
        float var  = (sqsum / count) - mean * mean;

        if (var <= 0.f)
            return;

        float stddev = std::sqrt(var);
        if (stddev <= 0.f)
            return;

        float psr = (max_val - mean) / stddev;

        // -----------------------------
        // 3. Angle weight
        // -----------------------------
        float angle_weight = std::max(psr - psr_threshold_, 0.f);
        if (angle_weight <= 0.f)
            return;

        // -----------------------------
        // 4. Translation (weighted average per push)
        // -----------------------------
        float weight_sum = 0.f;
        Vec3  weighted_t = {0.f,0.f,0.f};

        for (int i = 0; i < n; ++i) {
            float w = std::max(p_cc[i], 0.f);
            weight_sum += w;

            weighted_t.x += w * pts_[i].x;
            weighted_t.y += w * pts_[i].y;
            weighted_t.z += w * pts_[i].z;
        }

        if (weight_sum <= 0.f)
            return;

        weighted_t.x /= weight_sum;
        weighted_t.y /= weight_sum;
        weighted_t.z /= weight_sum;

        // -----------------------------
        // 5. Accumulate across angles
        // -----------------------------
        total_angle_weight_ += angle_weight;

        weighted_translation_sum_.x += angle_weight * weighted_t.x;
        weighted_translation_sum_.y += angle_weight * weighted_t.y;
        weighted_translation_sum_.z += angle_weight * weighted_t.z;

        // quaternion averaging
        Eigen::Quaternionf q(Rot);
        if (q.w() < 0.f)
            q.coeffs() *= -1.f;

        Eigen::Vector4f v = q.coeffs();
        weighted_rotation_sum_ += angle_weight * (v * v.transpose());
    }

    float get_cc() const
    {
        return total_angle_weight_;
    }

    Vec3 get_vec() const
    {
        if (total_angle_weight_ <= 0.f)
            return {0.f,0.f,0.f};

        return {
            weighted_translation_sum_.x / total_angle_weight_,
            weighted_translation_sum_.y / total_angle_weight_,
            weighted_translation_sum_.z / total_angle_weight_
        };
    }

    M33f get_rot() const
    {
        if (total_angle_weight_ <= 0.f)
            return M33f::Identity();

        Eigen::Matrix4f M =
            weighted_rotation_sum_ / total_angle_weight_;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4f> eig(M);

        Eigen::Quaternionf q(
            eig.eigenvectors().col(3)
            );

        return q.normalized().toRotationMatrix();
    }

    float get_dose() const
    {
        return 0.f;
    }
};

class CcTrackAlignmentGaussianFit {
private:
    const Vec3* pts_;
    int   n_pts_;
    int   n_ang_max_;
    float pix_size_;

    float max_radius_;   // grid extent

    float current_score_;
    float current_sigma_;
    Vec3  current_peak_;
    M33f  current_rot_;

public:

    CcTrackAlignmentGaussianFit(const Vec3* p_pts,int n_pts,int n_ang_max,float pix_size)
        : pts_(p_pts),
        n_pts_(n_pts),
        n_ang_max_(n_ang_max),
        pix_size_(pix_size)
    {
        compute_sigma_threshold();
        clear();
    }

    void compute_sigma_threshold()
    {
        max_radius_ = 0.f;

        for (int i = 0; i < n_pts_; ++i)
        {
            float r = sqrtf(pts_[i].x*pts_[i].x + pts_[i].y*pts_[i].y + pts_[i].z*pts_[i].z);
            if (r > max_radius_)
                max_radius_ = r;
        }
    }

    void clear()
    {
        current_score_ = 0.f;
        current_sigma_ = 0.f;
        current_peak_.x = 0;
        current_peak_.y = 0;
        current_peak_.z = 0;
        current_rot_.setIdentity();
    }

    void push(const float* p_cc,
              int n_pts,
              const M33f& Rot)
    {
        const int n = std::min(n_pts, n_pts_);
        if (n < 1) return;

        // If too few points, fallback to discrete max
        if (n < 6)
        {
            int imax = 0;
            float vmax = p_cc[0];

            for (int i = 1; i < n; ++i)
            {
                if (p_cc[i] > vmax)
                {
                    vmax = p_cc[i];
                    imax = i;
                }
            }

            if (vmax <= 0.f) return;

            float score = std::sqrt(vmax);

            if (score > current_score_)
            {
                current_score_ = score;
                current_peak_  = pts_[imax];
                current_rot_   = Rot;
                current_sigma_ = 0.f; // unknown
            }
            return;
        }

        // ---- Nonlinear Gaussian fit for n >= 6 ----

        // ---- Initial guesses ----
        float vmax = p_cc[0];
        float vmin = p_cc[0];
        int imax = 0;

        for (int i = 1; i < n; ++i)
        {
            if (p_cc[i] > vmax) { vmax = p_cc[i]; imax = i; }
            if (p_cc[i] < vmin) vmin = p_cc[i];
        }

        if (vmax <= 0.f) return;

        float A     = vmax - vmin;
        float c     = vmin;
        float sigma = max_radius_ / 3.f;

        V3f mu(pts_[imax].x,
               pts_[imax].y,
               pts_[imax].z);


        // ---- Levenberg–Marquardt iterations ----
        const int max_iter = 30;
        float lambda = 1e-2f;              // LM damping
        const float lambda_up = 10.f;
        const float lambda_down = 0.3f;

        // --------------------------------------------------
        // Levenberg–Marquardt iterations
        // --------------------------------------------------
        for (int iter = 0; iter < max_iter; ++iter)
        {
            Eigen::MatrixXf J(n,6);
            Eigen::VectorXf r(n);

            float cost = 0.f;

            for (int i = 0; i < n; ++i)
            {
                V3f x(pts_[i].x, pts_[i].y, pts_[i].z);
                V3f diff = x - mu;

                float r2 = diff.squaredNorm();
                float inv_sigma2 = 1.f / (sigma * sigma);

                float exp_term = std::exp(-0.5f * r2 * inv_sigma2);

                float model = A * exp_term + c;
                float ri = model - p_cc[i];

                r(i) = ri;
                cost += ri * ri;

                // ---- Jacobian ----

                J(i,0) = exp_term;

                // correct sign!
                J(i,1) = -A * exp_term * diff.x() * inv_sigma2;
                J(i,2) = -A * exp_term * diff.y() * inv_sigma2;
                J(i,3) = -A * exp_term * diff.z() * inv_sigma2;

                J(i,4) = A * exp_term * r2 / (sigma * sigma * sigma);

                J(i,5) = 1.f;
            }

            // Build LM system
            Eigen::MatrixXf H = J.transpose() * J;
            Eigen::VectorXf g = J.transpose() * r;

            H += lambda * H.diagonal().asDiagonal();

            Eigen::VectorXf delta = -H.ldlt().solve(g);
            // Eigen::VectorXf delta = -(H.colPivHouseholderQr().solve(g));
            // Eigen::VectorXf delta = -(H.completeOrthogonalDecomposition().solve(g));

            // Trial parameters
            float A_new     = A      + delta(0);
            float mux_new   = mu.x() + delta(1);
            float muy_new   = mu.y() + delta(2);
            float muz_new   = mu.z() + delta(3);
            float sigma_new = sigma  + delta(4);
            float c_new     = c      + delta(5);

            if (sigma_new <= 0.f || sigma_new > max_radius_ || A_new <= 0.f)
            {
                lambda *= lambda_up;
                continue;
            }

            // Compute new cost
            float new_cost = 0.f;
            for (int i = 0; i < n; ++i)
            {
                V3f x(pts_[i].x, pts_[i].y, pts_[i].z);
                V3f diff(x.x()-mux_new,
                         x.y()-muy_new,
                         x.z()-muz_new);

                float r2 = diff.squaredNorm();
                float exp_term = std::exp(-r2 / (2.f * sigma_new * sigma_new));
                float model = A_new * exp_term + c_new;

                float ri = model - p_cc[i];
                new_cost += ri * ri;
            }

            // Accept or reject
            if (new_cost < cost)
            {
                A     = A_new;
                mu.x() = mux_new;
                mu.y() = muy_new;
                mu.z() = muz_new;
                sigma = sigma_new;
                c     = c_new;

                lambda *= lambda_down;

                if (delta.norm() < 1e-6f)
                    break;
            }
            else
            {
                lambda *= lambda_up;
            }
        }

        // ---- Validity checks ----
        if (sigma <= 0.f || sigma > max_radius_) return;

        // ---- Radial projection if outside ----
        float mu_norm = mu.norm();
        if (mu_norm > max_radius_) mu *= (max_radius_ / mu_norm);

        float score = std::sqrt(A + c);

        if (score > current_score_)
        {
            current_score_  = score;
            current_peak_.x = mu(0);
            current_peak_.y = mu(1);
            current_peak_.z = mu(2);
            current_rot_    = Rot;
            current_sigma_  = sigma;
        }
    }

    float get_cc() const
    {
        return current_score_;
    }

    Vec3 get_vec() const
    {
        return current_peak_;
    }

    M33f get_rot() const
    {
        return current_rot_;
    }

    // ---- CryoEM B-factor / Dose ----
    float get_dose() const
    {
        if (current_sigma_ <= 0.f)
            return 0.f;

        float sigma_phys = current_sigma_ * pix_size_;

        return float(M_PI*M_PI) * sigma_phys * sigma_phys;
    }
};

class CcTrackerAlignment {
protected:
    using tracker_t = std::variant<CcTrackAlignmentMax,CcTrackAlignmentSigma,CcTrackAlignmentWgtAvg,CcTrackAlignmentGaussianFit>;
    tracker_t tracker;
protected:
    static tracker_t make_tracker(CcStatsType_t type,const Vec3* p_pts,int n_pts,const int n_ang_max,const float pix_size) {
        switch(type) {
            case CC_STATS_NONE:
                return CcTrackAlignmentMax(p_pts,n_pts,n_ang_max,pix_size);
            case CC_STATS_SIGMA:
                return CcTrackAlignmentSigma(p_pts,n_pts,n_ang_max,pix_size);
            case CC_STATS_WGT_AVG:
                return CcTrackAlignmentWgtAvg(p_pts,n_pts,n_ang_max,pix_size);
            case CC_STATS_GAUSSIAN_FIT:
                return CcTrackAlignmentGaussianFit(p_pts,n_pts,n_ang_max,pix_size);
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

