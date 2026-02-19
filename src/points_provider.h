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
        float h     = (step > 0.0f) ? step : 1.0f;
        float x_lim = h*floorf(fabsf(x_range)/h);
        float y_lim = h*floorf(fabsf(y_range)/h);
        float z_lim = h*floorf(fabsf(z_range)/h);
        
        if ((x_lim == 0) && (y_lim == 0) && (z_lim == 0)) {
            counter = 1;
            Vec3 *points = new Vec3[counter];
            points[0].x = 0;
            points[0].y = 0;
            points[0].z = 0;
            return points;
        }

        float x,y,z;
        
        counter=0;
        for(z=-z_lim; z<=z_lim; z=z+h) {
            for(y=-y_lim; y<=y_lim; y=y+h) {
                for(x=-x_lim; x<=x_lim; x=x+h) {
                    counter++;
                }
            }
        }

        Vec3 *points = new Vec3[counter];
        counter=0;
        for(z=-z_lim; z<=z_lim; z=z+h) {
            for(y=-y_lim; y<=y_lim; y=y+h) {
                for(x=-x_lim; x<=x_lim; x=x+h) {
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
        float h     = (step > 0.0f) ? step : 1.0f;
        float x_lim = h*floorf(fabsf(x_range)/h);
        float y_lim = h*floorf(fabsf(y_range)/h);
        float z_lim = h*floorf(fabsf(z_range)/h);
        
        if ((x_lim == 0) && (y_lim == 0) && (z_lim == 0)) {
            counter = 1;
            Vec3 *points = new Vec3[counter];
            points[0].x = 0;
            points[0].y = 0;
            points[0].z = 0;
            return points;
        }
        
        float A2 = x_lim * x_lim;
        float B2 = y_lim * y_lim;
        float C2 = z_lim * z_lim;
        
        float x,y,z,kx,ky,kz,X2,Y2,Z2,R2;
        
        kx = (A2 > 0.0f) ? 1.0f : 0.0f;
        ky = (B2 > 0.0f) ? 1.0f : 0.0f;
        kz = (C2 > 0.0f) ? 1.0f : 0.0f;
        
        if (A2 > 0.0f) {
            ky *= A2;
            kz *= A2;
        }
        if (B2 > 0.0f) {
            kx *= B2;
            kz *= B2;
        }
        if (C2 > 0.0f) {
            kx *= C2;
            ky *= C2;
        }
        
        R2 = 1.0f;
        if (A2 > 0.0f) R2 *= A2;
        if (B2 > 0.0f) R2 *= B2;
        if (C2 > 0.0f) R2 *= C2;
        
        // (x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1)  --> (x^2*b^2*c^2 + y^2*a^2*c^2 + z^2*a^2*b^2 <= a^2*b^2*c^2)
        //but if any or a, b,or c, they simply get excluded from equation of the N-k dimentional ellipsoid
        
        counter=0;
        for(z=-z_lim; z<=z_lim; z=z+h) {
            Z2 = z * z * kz;           
            for(y=-y_lim; y<=y_lim; y=y+h) {
                Y2 = y * y * ky;           
                for(x=-x_lim; x<=x_lim; x=x+h) {
                    X2 = x * x * kx;               
                    if( (X2 + Y2 + Z2) <= R2 ) {
                        counter++;
                    }
                }
            }
        }

        Vec3 *points = new Vec3[counter];
        counter=0;
        for(z=-z_lim; z<=z_lim; z=z+h) {
            Z2 = z * z * kz;
            for(y=-y_lim; y<=y_lim; y=y+h) {
                Y2 = y * y * ky; 
                for(x=-x_lim; x<=x_lim; x=x+h) {
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
        float h     = (step > 0.0f) ? step : 1.0f;
        float x_lim = h*floorf(fabsf(x_range)/h);
        float y_lim = h*floorf(fabsf(y_range)/h);
        float z_lim = h*floorf(fabsf(z_range)/h);
        
        if ((x_lim == 0) && (y_lim == 0) && (z_lim == 0)) {
            counter = 1;
            Vec3 *points = new Vec3[counter];
            points[0].x = 0;
            points[0].y = 0;
            points[0].z = 0;
            return points;
        }

        float A2 = x_lim * x_lim;
        float B2 = y_lim * y_lim;
        
        float x,y,z,kx,ky,X2,Y2,R2;
        
        kx = (A2 > 0.0f) ? 1.0f : 0.0f;
        ky = (B2 > 0.0f) ? 1.0f : 0.0f;
        
        if (A2 > 0.0f) ky *= A2;
        if (B2 > 0.0f) kx *= B2;
        
        R2 = 1.0f;
        if (A2 > 0.0f) R2 *= A2;
        if (B2 > 0.0f) R2 *= B2;
        
        // (x^2/a^2 + y^2/b^2 <= 1) --> (x^2*b^2 + y^2*a^2 = a^2*b^2), but without devision and sqrt
        counter=0;
        for(z=-z_lim; z<=z_lim; z=z+h) {
            for(y=-y_lim; y<=y_lim; y=y+h) {
                Y2 = y * y * ky;
                for(x=-x_lim; x<=x_lim; x=x+h) {
                    X2 = x * x * kx;
                    if( (X2 + Y2) <= R2 ) {
                        counter++;
                    }
                }
            }
        }

        Vec3 *points = new Vec3[counter];
        counter=0;
        for(z=-z_lim; z<=z_lim; z=z+h) {
            for(y=-y_lim; y<=y_lim; y=y+h) {
                Y2 = y * y * ky;
                for(x=-x_lim; x<=x_lim; x=x+h) {
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