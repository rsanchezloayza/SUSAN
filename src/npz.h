/*
 * This file is part of the Substack Analysis (SUSAN) framework.
 * Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
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

#ifndef NPZ_H
#define NPZ_H

/// NumPy .npz writer: a zip archive of .npy entries, stored without compression (numpy.load reads it).

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>

namespace Npz {

inline uint32_t crc32(const uint8_t*d,size_t n,uint32_t c=0) {
    static uint32_t tbl[256]; static bool init=false;
    if( !init ) {
        for(uint32_t i=0;i<256;i++) { uint32_t r=i; for(int k=0;k<8;k++) r = (r&1) ? (0xEDB88320u^(r>>1)) : (r>>1); tbl[i]=r; }
        init=true;
    }
    c = ~c;
    for(size_t i=0;i<n;i++) c = tbl[(c^d[i])&0xFF]^(c>>8);
    return ~c;
}

struct Entry {
    std::string name;
    std::vector<uint8_t> bytes;
};

/// One array: descr is the numpy type string ("<f4", "<i4", "<U8", ...), shape {} for a scalar.
inline Entry npy(const std::string&name,const std::string&descr,const std::vector<size_t>&shape,const void*data,size_t nbytes) {
    std::string sh="(";
    for(size_t i=0;i<shape.size();i++) { sh += std::to_string(shape[i]); sh += (shape.size()==1 || i+1<shape.size()) ? "," : ""; if(i+1<shape.size()) sh+=" "; }
    sh += ")";
    std::string hdr = "{'descr': '"+descr+"', 'fortran_order': False, 'shape': "+sh+", }";
    size_t total = 10 + hdr.size() + 1;
    size_t pad = (64 - total%64)%64;
    hdr += std::string(pad,' ') + "\n";
    Entry e;
    e.name = name+".npy";
    const uint8_t magic[8] = {0x93,'N','U','M','P','Y',1,0};
    e.bytes.insert(e.bytes.end(),magic,magic+8);
    uint16_t hl = (uint16_t)hdr.size();
    e.bytes.push_back(hl&0xFF); e.bytes.push_back(hl>>8);
    e.bytes.insert(e.bytes.end(),hdr.begin(),hdr.end());
    const uint8_t*p=(const uint8_t*)data;
    e.bytes.insert(e.bytes.end(),p,p+nbytes);
    return e;
}

inline void put16(std::vector<uint8_t>&b,uint16_t v){ b.push_back(v&0xFF); b.push_back(v>>8); }
inline void put32(std::vector<uint8_t>&b,uint32_t v){ for(int i=0;i<4;i++) b.push_back((v>>(8*i))&0xFF); }

inline bool write(const char*fname,const std::vector<Entry>&ents) {
    std::vector<uint8_t> out, cd;
    for(const Entry&e: ents) {
        uint32_t crc = crc32(e.bytes.data(),e.bytes.size());
        uint32_t off = (uint32_t)out.size();
        put32(out,0x04034b50); put16(out,20); put16(out,0); put16(out,0); put16(out,0); put16(out,0x21);
        put32(out,crc); put32(out,(uint32_t)e.bytes.size()); put32(out,(uint32_t)e.bytes.size());
        put16(out,(uint16_t)e.name.size()); put16(out,0);
        out.insert(out.end(),e.name.begin(),e.name.end());
        out.insert(out.end(),e.bytes.begin(),e.bytes.end());
        put32(cd,0x02014b50); put16(cd,20); put16(cd,20); put16(cd,0); put16(cd,0); put16(cd,0); put16(cd,0x21);
        put32(cd,crc); put32(cd,(uint32_t)e.bytes.size()); put32(cd,(uint32_t)e.bytes.size());
        put16(cd,(uint16_t)e.name.size()); put16(cd,0); put16(cd,0); put16(cd,0); put16(cd,0); put32(cd,0); put32(cd,off);
        cd.insert(cd.end(),e.name.begin(),e.name.end());
    }
    uint32_t cd_off = (uint32_t)out.size();
    out.insert(out.end(),cd.begin(),cd.end());
    put32(out,0x06054b50); put16(out,0); put16(out,0); put16(out,(uint16_t)ents.size()); put16(out,(uint16_t)ents.size());
    put32(out,(uint32_t)cd.size()); put32(out,cd_off); put16(out,0);
    FILE*fp = fopen(fname,"wb");
    if( fp == NULL ) return false;
    bool ok = fwrite(out.data(),1,out.size(),fp) == out.size();
    fclose(fp);
    return ok;
}

}

#endif /// NPZ_H
