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

#ifndef POINTS_PROVIDER_H
#define POINTS_PROVIDER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "datatypes.h"

using namespace Eigen;

class PointsProvider {

public:
    static Vec3 *cuboid(uint32&counter, const float x_range, const float y_range, const float z_range, const float step) {
        float h  = (step > 0.0f) ? step : 1.0f;
        int   nx = (int)floorf(fabsf(x_range)/h);
        int   ny = (int)floorf(fabsf(y_range)/h);
        int   nz = (int)floorf(fabsf(z_range)/h);

        if ((nx == 0) && (ny == 0) && (nz == 0)) {
            counter = 1;
            Vec3 *points = new Vec3[counter];
            points[0].x = 0;
            points[0].y = 0;
            points[0].z = 0;
            return points;
        }

        counter = (uint32)(2*nz+1) * (uint32)(2*ny+1) * (uint32)(2*nx+1);
        Vec3 *points = new Vec3[counter];
        counter=0;
        for(int iz=-nz; iz<=nz; iz++) {
            float z = iz * h;
            for(int iy=-ny; iy<=ny; iy++) {
                float y = iy * h;
                for(int ix=-nx; ix<=nx; ix++) {
                    float x = ix * h;
                    points[counter].x = x;
                    points[counter].y = y;
                    points[counter].z = z;
                    counter++;
                }
            }
        }

        return points;
    }
    
    static Vec3 *ellipsoid(uint32&counter, const float x_range, const float y_range, const float z_range, const float step) {
        float h  = (step > 0.0f) ? step : 1.0f;
        int   nx = (int)floorf(fabsf(x_range)/h);
        int   ny = (int)floorf(fabsf(y_range)/h);
        int   nz = (int)floorf(fabsf(z_range)/h);

        if ((nx == 0) && (ny == 0) && (nz == 0)) {
            counter = 1;
            Vec3 *points = new Vec3[counter];
            points[0].x = 0;
            points[0].y = 0;
            points[0].z = 0;
            return points;
        }

        float x_lim = nx * h;
        float y_lim = ny * h;
        float z_lim = nz * h;

        float A2 = x_lim * x_lim;
        float B2 = y_lim * y_lim;
        float C2 = z_lim * z_lim;

        float kx,ky,kz,X2,Y2,Z2,R2;

        // Multiply x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1 through by a^2*b^2*c^2 to avoid
        // division: x^2*b^2*c^2 + y^2*a^2*c^2 + z^2*a^2*b^2 <= a^2*b^2*c^2.
        // For a degenerate axis (lim == 0) its term is dropped and R2 only accumulates
        // the non-zero axes, reducing the shape to a lower-dimensional ellipsoid.
        kx = (A2 > 0.0f) ? 1.0f : 0.0f;
        ky = (B2 > 0.0f) ? 1.0f : 0.0f;
        kz = (C2 > 0.0f) ? 1.0f : 0.0f;

        if (A2 > 0.0f) { ky *= A2; kz *= A2; }
        if (B2 > 0.0f) { kx *= B2; kz *= B2; }
        if (C2 > 0.0f) { kx *= C2; ky *= C2; }

        R2 = 1.0f;
        if (A2 > 0.0f) R2 *= A2;
        if (B2 > 0.0f) R2 *= B2;
        if (C2 > 0.0f) R2 *= C2;

        counter=0;
        for(int iz=-nz; iz<=nz; iz++) {
            float z = iz * h;
            Z2 = z * z * kz;
            for(int iy=-ny; iy<=ny; iy++) {
                float y = iy * h;
                Y2 = y * y * ky;
                for(int ix=-nx; ix<=nx; ix++) {
                    float x = ix * h;
                    X2 = x * x * kx;
                    if( (X2 + Y2 + Z2) <= R2 ) {
                        counter++;
                    }
                }
            }
        }

        Vec3 *points = new Vec3[counter];
        counter=0;
        for(int iz=-nz; iz<=nz; iz++) {
            float z = iz * h;
            Z2 = z * z * kz;
            for(int iy=-ny; iy<=ny; iy++) {
                float y = iy * h;
                Y2 = y * y * ky;
                for(int ix=-nx; ix<=nx; ix++) {
                    float x = ix * h;
                    X2 = x * x * kx;
                    if( (X2 + Y2 + Z2) <= R2 ) {
                        points[counter].x = x;
                        points[counter].y = y;
                        points[counter].z = z;
                        counter++;
                    }
                }
            }
        }
        return points;
    }
    
    static Vec3 *cylinder(uint32&counter, const float x_range, const float y_range, const float z_range, const float step) {
        float h  = (step > 0.0f) ? step : 1.0f;
        int   nx = (int)floorf(fabsf(x_range)/h);
        int   ny = (int)floorf(fabsf(y_range)/h);
        int   nz = (int)floorf(fabsf(z_range)/h);

        if ((nx == 0) && (ny == 0) && (nz == 0)) {
            counter = 1;
            Vec3 *points = new Vec3[counter];
            points[0].x = 0;
            points[0].y = 0;
            points[0].z = 0;
            return points;
        }

        float x_lim = nx * h;
        float y_lim = ny * h;

        float A2 = x_lim * x_lim;
        float B2 = y_lim * y_lim;

        float kx,ky,X2,Y2,R2;

        kx = (A2 > 0.0f) ? 1.0f : 0.0f;
        ky = (B2 > 0.0f) ? 1.0f : 0.0f;

        if (A2 > 0.0f) ky *= A2;
        if (B2 > 0.0f) kx *= B2;

        R2 = 1.0f;
        if (A2 > 0.0f) R2 *= A2;
        if (B2 > 0.0f) R2 *= B2;

        // (x^2/a^2 + y^2/b^2 <= 1) --> (x^2*b^2 + y^2*a^2 <= a^2*b^2), without division
        counter=0;
        for(int iz=-nz; iz<=nz; iz++) {
            for(int iy=-ny; iy<=ny; iy++) {
                float y = iy * h;
                Y2 = y * y * ky;
                for(int ix=-nx; ix<=nx; ix++) {
                    float x = ix * h;
                    X2 = x * x * kx;
                    if( (X2 + Y2) <= R2 ) {
                        counter++;
                    }
                }
            }
        }

        Vec3 *points = new Vec3[counter];
        counter=0;
        for(int iz=-nz; iz<=nz; iz++) {
            float z = iz * h;
            for(int iy=-ny; iy<=ny; iy++) {
                float y = iy * h;
                Y2 = y * y * ky;
                for(int ix=-nx; ix<=nx; ix++) {
                    float x = ix * h;
                    X2 = x * x * kx;
                    if( (X2 + Y2) <= R2 ) {
                        points[counter].x = x;
                        points[counter].y = y;
                        points[counter].z = z;
                        counter++;
                    }
                }
            }
        }
        return points;
    }
    
    static Vec3 *rectangle(uint32&counter, const float x_range, const float y_range, const float step) {
        return cuboid(counter, x_range, y_range, 0.0f, step);
    }
    static Vec3 *sphere(uint32&counter, const float radius, const float step) {
        return ellipsoid(counter, radius, radius, radius, step);
    }
    static Vec3 *circle(uint32&counter, const float x_range, const float y_range, const float step) {
        return ellipsoid(counter, x_range, y_range, 0.0f, step);
    }
};
#endif /// POINTS_PROVIDER_H