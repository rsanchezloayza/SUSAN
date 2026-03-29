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

#ifndef GPU_FFT_H
#define GPU_FFT_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "datatypes.h"

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cufft.h"

namespace GpuFFT {

class XFFT_base {
protected:
    cufftHandle handler;

public:
    XFFT_base() {
        handler = 0;
    }

    ~XFFT_base() {
        if( handler != 0 )
            cufftDestroy(handler);
    }
    
    void set_stream(cudaStream_t strm) {
        cufftResult err = cufftSetStream(handler, strm);
        if( err != CUFFT_SUCCESS ) {
            fprintf(stderr,"Error setting CUFFT stream [Code error: %d].\n",err);
            exit(1);
        }
    }
    
};

class FFT1D : public XFFT_base {
public:
    void alloc(const int X, const int Y) {
        int rank = 1;
        int m[1] = {X};
        int idist = X;
        int odist = (X/2)+1;
        int inembed[] = {X};
        int onembed[] = {(X/2)+1};
        int istride = 1;
        int ostride = 1;

        if ( cufftPlanMany(&handler, rank, m, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, Y) != CUFFT_SUCCESS ) {
            fprintf(stderr,"Error allocating forward FFT1D batch.\n");
            exit(1);

        }
    }
    
    void exec(float2*g_out, float*g_in) {
        cufftExecR2C(handler,g_in,g_out);
    }
};

class FFT1D_full : public XFFT_base {
public:
    void alloc(const int X, const int Y) {
        int rank = 1;
        int m[1] = {X};
        int idist = X;
        int odist = X;
        int inembed[] = {X};
        int onembed[] = {X};
        int istride = 1;
        int ostride = 1;

        if ( cufftPlanMany(&handler, rank, m, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, Y) != CUFFT_SUCCESS ) {
            fprintf(stderr,"Error allocating forward FFT1D batch.\n");
            exit(1);

        }
    }

    void exec(float2*g_out, float2*g_in) {
        cufftExecC2C(handler,g_in,g_out,CUFFT_FORWARD);
    }
};

class IFFT1D_full : public XFFT_base {
public:
    void alloc(const int X, const int Y) {
        int rank = 1;
        int m[1] = {X};
        int idist = X;
        int odist = X;
        int inembed[] = {X};
        int onembed[] = {X};
        int istride = 1;
        int ostride = 1;

        if ( cufftPlanMany(&handler, rank, m, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, Y) != CUFFT_SUCCESS ) {
            fprintf(stderr,"Error allocating inverse FFT1D batch.\n");
            exit(1);

        }
    }

    void exec(float2*g_out, float2*g_in) {
        cufftExecC2C(handler,g_in,g_out,CUFFT_INVERSE);
    }
};

class FFT2D : public XFFT_base {
public:
    void alloc(const int X, const int Y, const int Z) {
        int rank = 2;
        int m[2] = {Y, Y};
        int idist = Y*Y;
        int odist = X*Y;
        int inembed[] = {Y, Y};
        int onembed[] = {Y, X};
        int istride = 1;
        int ostride = 1;

        if ( cufftPlanMany(&handler, rank, m, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, Z) != CUFFT_SUCCESS ) {
            fprintf(stderr,"Error allocating forward FFT2D batch.\n");
            exit(1);

        }
    }

    void exec(float2*g_out, float*g_in) {
        cufftExecR2C(handler,g_in,g_out);
    }
};

class IFFT2D : public XFFT_base {
public:
    void alloc(const int X, const int Y, const int Z) {
        int rank = 2;
        int m[2] = {Y, Y};
        int odist = Y*Y;
        int idist = X*Y;
        int onembed[] = {Y, Y};
        int inembed[] = {Y, X};
        int istride = 1;
        int ostride = 1;

        if ( cufftPlanMany(&handler, rank, m, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, Z) != CUFFT_SUCCESS ) {
            fprintf(stderr,"Error allocating inverse FFT2D batch.\n");
            exit(1);

        }
    }

    void exec(float*g_out, float2*g_in) {
        cufftExecC2R(handler,g_in,g_out);
    }
};

class FFT2D_full : public XFFT_base {
public:
    void alloc(const int X, const int Y, const int Z) {
        int rank = 2;
        int m[2] = {Y, Y};
        int idist = Y*Y;
        int odist = X*Y;
        int inembed[] = {Y, Y};
        int onembed[] = {Y, X};
        int istride = 1;
        int ostride = 1;

        if ( cufftPlanMany(&handler, rank, m, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, Z) != CUFFT_SUCCESS ) {
            fprintf(stderr,"Error allocating forward FFT2D batch.\n");
            exit(1);

        }
    }

    void exec(float2*g_out, float2*g_in) {
        cufftExecC2C(handler,g_in,g_out,CUFFT_FORWARD);
    }
};

class FFT3D : public XFFT_base {
public:
    void alloc(const int N) {

        cufftResult err = cufftPlan3d(&handler, N, N, N, CUFFT_R2C);
        if( err != CUFFT_SUCCESS ) {
            fprintf(stderr,"Error allocating forward FFT3D [Code error: %d].\n",err);
            exit(1);
        }
    }

    void exec(float2*g_out, float*g_in) {
        cufftExecR2C(handler,g_in,g_out);
    }
};

class IFFT3D : public XFFT_base {
public:
    void alloc(const int N) {

        cufftResult err = cufftPlan3d(&handler, N, N, N, CUFFT_C2R );
        if( err != CUFFT_SUCCESS ) {
            fprintf(stderr,"Error allocating inverse FFT3D [Code error: %d].\n",err);
            exit(1);
        }
    }

    void exec(float*g_out, float2*g_in) {
        cufftExecC2R(handler,g_in,g_out);
    }
};


}

#endif /// GPU_FFT_H

