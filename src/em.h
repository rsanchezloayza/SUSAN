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

#ifndef EM_H
#define EM_H

#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <string>
#include "datatypes.h"

using namespace std;

namespace EM {
    void write(const single *data, const uint32 X, const uint32 Y, const uint32 Z, const char*mapname) {
        FILE*fp=fopen(mapname,"wb");
        uint32 header[128];
        uint8 *p8 = (uint8*)(header);
        for(int i=0;i<128;i++) header[i] = 0;
        p8[0] = 6;
        p8[3] = 5;
        header[1]  = X;
        header[2]  = Y;
        header[3]  = Z;
        fwrite((void*)header,sizeof(uint32),128,fp);
        fwrite((void*)data,sizeof(single),(size_t)X*Y*Z,fp);
        fclose(fp);
    }

}

#endif 

