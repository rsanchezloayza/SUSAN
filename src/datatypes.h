/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2022 Ricardo Miguel Sanchez Loayza.
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

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

typedef uint8_t  uint8;
typedef int32_t  int32;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef float    single;

#define SUSAN_CARRIER_RETURN '\r'
//#define SUSAN_CARRIER_RETURN 0

#define SUSAN_FILENAME_LENGTH 1024
#define SUSAN_CHUNK_SIZE (1024*1024)
#define SUSAN_FLOAT_TOL 5e-8
#define SUSAN_MAX_N_GPU 24
#define SUSAN_CUDA_THREADS 1024
#define SUSAN_CUDA_WARP 32

#define RAD2DEG 57.295779513082323
#define DEG2RAD  0.017453292519943

#define SUSAN_SVG_SHADOW_BG "#E6E6E6"
//#define SUSAN_SVG_FG_A "#FDB366"
#define SUSAN_SVG_FG_A "#F67E4B"
//#define SUSAN_SVG_FG_B "#98CAE1"
#define SUSAN_SVG_FG_B "#6EA6CD"
//#define SUSAN_SVG_FG_C "#EAECCC"
#define SUSAN_SVG_FG_C "#CFD48A"
#define SUSAN_SVG_FG_D "#B8578F"

typedef struct {
    float x;
    float y;
} Vec2;

typedef struct {
    float x;
    float y;
    float z;
} Vec3;

typedef struct {
    uint32 x;
    uint32 y;
    uint32 z;
} VUInt3;

typedef struct {
    float xx;
    float xy;
    float xz;
    float yx;
    float yy;
    float yz;
    float zx;
    float zy;
    float zz;
} Rot33; /// Row major

typedef struct {
	uint32  ptcl_id;
    uint32  tomo_id;
    uint32  tomo_cix_deprecated; /// Cached index of tomo_id in the tomograms.
                                /// Not read anymore (it goes stale when the
                                /// particles are paired with another tomostxt):
                                /// use Tomograms::get_cix(tomo_id). Kept to
                                /// preserve the ptclsraw record layout.
    Vec3    pos;      /// Angstroms
    uint32  ref_cix;
    uint32  half_id;
    single  extra_1;
    single  extra_2;
} PtclInf;

typedef struct {
    float U;
    float V;
    float angle;   /// Sexagesimal degrees
    float ph_shft; /// Radians
    float Bfactor;
    float ExpFilt;
    float max_res;
    float score;
} Defocus;

typedef struct {
    float CsLambda3PiH;
    float LambdaPi;
    float AC;
    float CA;
    float apix;
} CtfConst;

typedef struct {
    float fp_min;
    float fp_max;
    float fp_rol;
} Bandpass_t;

typedef struct {
    Rot33 R;
    Vec3  t;
    float w;
} Proj2D;

typedef struct {
    float fill;
    bool  has_fill;
    float gat_a;
    float gat_b;
} ProjStats;

typedef enum {
	DONE  = -1,
    EMPTY = 0,
    READY = 1
} Status_t;

typedef struct {
    float U;
    float V;
    float angle_deg;
    float phase_shift_rad;
} DefocusDelta;

typedef enum {
    VERBOSITY_MINIMAL = 0,
    VERBOSITY_BASIC,
    VERBOSITY_FULL
} Verbosity_t;

typedef enum {
    NO_NORM=0,
    ZERO_MEAN,
    ZERO_MEAN_1_STD,
    GAT_RAW,
    GAT_NORMAL,
    LOCAL_NORM
} NormalizationType_t;

typedef enum {
    WGT_NONE=0,
    WGT_3D,
    WGT_2D,
    WGT_3DCC,
    WGT_2DCC
} WeightingType_t;

typedef enum {
    PAD_ZERO=0,
    PAD_GAUSSIAN
} PaddingType_t;

typedef enum {
    ALI_CTF_DISABLED=0,
    ALI_CTF_PHASE_FLIP,
    ALI_CTF_ON_REFERENCE,
    ALI_CTF_ON_SUBSTACK,
    ALI_CTF_ON_SUBSTACK_SSNR
} CtfAlignmentType_t;

typedef enum {
    CC_TYPE_BASIC=0,
    CC_TYPE_CFSC,
    CC_TYPE_CFSC_SUBSTACK
} CcType_t;

inline bool cc_type_whitens_substack(const uint32 cc_type) {
    return (cc_type == CC_TYPE_CFSC) || (cc_type == CC_TYPE_CFSC_SUBSTACK);
}

inline bool cc_type_whitens_reference(const uint32 cc_type) {
    return (cc_type == CC_TYPE_CFSC);
}

typedef enum {
    CC_STATS_NONE=0,
    CC_STATS_SIGMA
} CcStatsType_t;

#define W_INV_STD_LINEAR 0.41f
#define W_INV_STD_KB     0.69f
#define GRID_CORRECT_KB_MAX 4.0f
#define EDGE_ERODE          1
#define EDGE_MIN_VALID      16
#define SUSAN_LCN_PROFILE_FRAC 0.05f  /// reference radius: outermost shell with rms above this fraction of the maximum
#define SUSAN_LCN_MARGIN       1.2f   /// local window half-side as a multiple of the reference radius
#define SUSAN_LCN_NOREF_FRAC   0.5f   /// local window half-side as a fraction of the box when there is no reference
#define FILL_MIN_RUN        8
#define FILL_TOL            1e-5f
#define GAT_TILE            32
#define GAT_MAX_TILES       256
#define GAT_MIN_TILES       16
#define GAT_RAW_A_TOL       0.25f
#define GAT_RAW_B_TOL       1.0f
#define SUSAN_MPI_CHUNK_BYTES ((uint64_t)1<<30)   /// bytes per MPI message of the byte-buffer transfers (MPI counts are int)
#define ALI_STATS_ANG_MAX_GS 2   /// alignment_stats: the groups up to this size keep the CC per scanned angle
#define SUSAN_CTF_STATS_BINS 512    /// refinement_stats profiles: bins of the linearized coordinate v = s^2 D(phi) / D_ref
#define SUSAN_CTF_STATS_VMAX 1.3f   /// profile range in units of the band's maximum s^2 (D(phi) / D_ref up to 1.3)
#define W_INV_STD_SPLAT     0.74f
#define NOISE_RING_GAP      3.0f
#define NOISE_RING_SPAN     2.0f
#define NOISE_RING_MIN_SPAN 4.0f
#define NOISE_RING_EDGE     2.0f
#define NOISE_RING_MAX_PTS  1024
/// Angular-spread splatting (see splat_sigma_t / splat_radius in gpu_kernel_vol.h).
#define SPLAT_SIGMA_0       0.4f    /// closest match to trilinear over all sub-voxel offsets
#define SPLAT_SIGMA_MAX     2.0f
#define SPLAT_RAD_MAX       4       /// 9x9x9
#define SPLAT_RAD_K         2.2f
#define SPLAT_W_REL         2e-2f   /// relative tap cull (of the kernel's own peak tap)
#define SPLAT_WARPS         8       /// source pixels per block in the main kernel (occupancy only)

typedef enum {
    GRIDDING_LINEAR_FWD=0,
    GRIDDING_KAISER_BESSEL_FWD,
} GriddingType_t;

typedef enum {
    INV_NO_INV=0,
    INV_PHASE_FLIP,
    INV_WIENER,
    INV_WIENER_SSNR,
    INV_WIENER_ARCTAN,
    INV_WIENER_LOGISTIC,
    INV_PRE_WIENER,
} CtfInversionType_t;

typedef enum {
    PRJ_NO_CTF=0,
    PRJ_STANDARD
} CtfProjectionType_t;

typedef enum {
    ASTIG_NONE=0,
    ASTIG_MAGNITUDE,   /// U-V per series, angle per projection
    ASTIG_SERIES,
    ASTIG_PROJECTION
} CtfAstigmatismType_t;

typedef enum {
    ALI_OUTPUT_ALIGNMENT=0,
    ALI_OUTPUT_DENSE,
    ALI_OUTPUT_ALIGNMENT_STATS   /// alignment + the 2D statistics (<output stem>.stats.npz), type 2
} AliOutputType_t;

typedef enum {
    CTF_OUTPUT_REFINEMENT=0,
    CTF_OUTPUT_REFINEMENT_STATS   /// particles unchanged + the CTF statistics (<output stem>.ctfstats.npz)
} CtfOutputType_t;

typedef enum {
    DENSE_REDUCE_MAX=0,
    DENSE_REDUCE_MEAN
} DenseReduce_t;

typedef enum {
    ELLIPSOID,
    CYLINDER,
    CIRCLE,
    CUBOID,
    RECTANGLE,
} OffsetType_t;

typedef enum {
    TOMOGRAM_SPACE,
    REFERENCE_SPACE,
} OffsetSpace_t;

typedef enum {
    CROP_MRC,
    CROP_EM,
} CropFormat_t;


