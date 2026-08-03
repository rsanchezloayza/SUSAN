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

#ifndef IO_H
#define IO_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <sstream>
#include <unordered_map>
#include <sys/stat.h>
#include <ftw.h>
#include <unistd.h>

#include <errno.h>

#include "datatypes.h"

namespace IO {

////

bool exists(const char*filename) {
    struct stat buffer;
    return ( stat( filename, &buffer) == 0 );
}

bool exist_dir(const char*dirname) {
    struct stat buffer;
    return ( stat( dirname, &buffer) == 0 ) && ( S_ISDIR(buffer.st_mode) );
}

int unlink_cb(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf) {
    int rv = remove(fpath);
    if( rv ) perror(fpath);
    return rv;
}

void create_dir(const char*filename) {
    if( !exist_dir(filename) )
        mkdir(filename, 0755);
}

void delete_dir(const char*filename) {
    if( exists(filename) )
        nftw(filename, unlink_cb, 64, FTW_DEPTH | FTW_PHYS);
}

void get_dirname(char*out,const char*path) {
    const char*sep = strrchr(path,'/');
    if( sep == NULL ) {
        strcpy(out,".");
    }
    else {
        size_t n = sep - path;
        if( n == 0 ) n = 1; /// path was "/file", keep the root slash
        memcpy(out,path,n);
        out[n] = 0;
    }
}

bool check_file_extension(const char*filename,const char*extension) {

    int fn_len  = strlen(filename);
    int ext_len = strlen(extension);

    return (strcmp( filename-ext_len+fn_len, extension )==0);

}

////

bool check_fread(void *ptr, size_t size, size_t nmemb, FILE *fp) {
    size_t read_nmemb = fread(ptr,size,nmemb,fp);
    if( read_nmemb != nmemb ) {
        fprintf(stderr,"Error reading file: %ld/%ld (%ld) [%d/%d]\n",read_nmemb,nmemb,size,ferror(fp),errno);
        return false;
    }
    return true;
}

////

namespace DefocusIO {

    bool read(Defocus&def,FILE*fp) {
        int n = 0;
        n += fscanf(fp,"%f",&def.U);
        n += fscanf(fp,"%f",&def.V);
        n += fscanf(fp,"%f",&def.angle);
        n += fscanf(fp,"%f",&def.ph_shft);
        n += fscanf(fp,"%f",&def.Bfactor);
        n += fscanf(fp,"%f",&def.ExpFilt);
        n += fscanf(fp,"%f",&def.max_res);
        n += fscanf(fp,"%f",&def.score);
        return (n==8);
    }

    void write(FILE*fp,const Defocus&def) {
        fprintf(fp,"%10.2f  %10.2f  %8.3f  %8.3f  %8.2f  %8.2f  %6.3f  %e\n",def.U,def.V,def.angle,def.ph_shft,def.Bfactor,def.ExpFilt,def.max_res,def.score);
    }

}

////

class TxtParser {

protected:
    char buffer[ SUSAN_FILENAME_LENGTH ];
    FILE *fp;
    std::unordered_map<std::string, std::string> args;

public:
    TxtParser(const char*filename,const char*extension) {
        if( IO::exists( filename ) ) {
            if( IO::check_file_extension(filename,extension) ) {
                fp = fopen(filename,"r");
            }
            else {
                fprintf(stderr,"File %s: wrong extension (%s).\n",filename,extension);
                exit(0);
            }
        }
        else {
            fprintf(stderr,"File %s does not exist.\n",filename);
            exit(0);
        }
    }

    ~TxtParser() {
        fclose(fp);
    }

    const char*get_buffer() {
        return (const char*)buffer;
    }

    void get_value(uint32&value,const char*tag) {
        int tmp;
        get_value(tmp,tag);
        value = (uint32)tmp;
    }

    void get_value(int&value,const char*tag) {
        value = atoi( get_char_ptr(tag) );
    }

    void get_value(float&value,const char*tag) {
        value = atof( get_char_ptr(tag) );
    }

    void get_value(VUInt3&value,const char*tag) {
        char*ptr = get_char_ptr(tag);
        remove_commas(ptr);
        if( sscanf(ptr,"%d %d %d",&value.x,&value.y,&value.z) != 3 ) {
            fprintf(stderr,"Error parsing %s: %s\n",tag,ptr);
        }
    }

    void get_value(Vec3&value,const char*tag) {
        char*ptr = get_char_ptr(tag);

        if( sscanf(ptr,"%f %f %f",&value.x,&value.y,&value.z) != 3 ) {
            fprintf(stderr,"Error parsing %s: %s\n",tag,ptr);
        }
    }

    void get_str(char*str,const char*tag) {
        char*ptr = get_char_ptr(tag);
        if( ptr[0] > 0 )
            strcpy(str,ptr);
        else
            str[0] = 0;
    }

    char*read_line_raw() {
        get_line();
        return buffer;
    }

    bool get_next_line() {
        bool should_continue = true;

        while( should_continue ) {
            if( fgets(buffer,SUSAN_FILENAME_LENGTH,fp) == NULL )
                return false;
            else {
                int buf_len = strlen(buffer);
                if( buf_len   ==  0  ) continue;
                if( buffer[0] == '#' ) return false;

                if( buffer[buf_len-1] == '\n' )
                    buffer[buf_len-1] = 0;

                should_continue = false;
                return true;
            }
        }

        return true;
    }

    void parse_args() {
        args.clear();

        while( fgets(buffer,SUSAN_FILENAME_LENGTH,fp) ) {
            int buf_len = strlen(buffer);
            if( buf_len   ==  0  ) continue;
            if( buffer[0] == '#' ) break;

            std::string line(buffer);
            line.erase(line.find_last_not_of("\r\n")+1);

            size_t pos = line.find(':');
            if (pos == std::string::npos) {
                fprintf(stderr,"Truncated file\n");
                exit(1);
            }

            std::string key = line.substr(0,pos);
            std::string val = line.substr(pos+1);

            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            val.erase(0, val.find_first_not_of(" \t"));
            val.erase(val.find_last_not_of(" \t") + 1);


            args[key] = val;
        }
    }

    bool args_get_val(uint32&val,const char*tag,const uint32 def_val=0) {
        std::string key(tag);
        val = def_val;
        if( !arg_exists(key) ) return false;
        val = static_cast<uint32_t>(std::stoul(args[key]));
        return true;
    }

    bool args_get_val(int&val,const char*tag,const int def_val=0) {
        std::string key(tag);
        val = def_val;
        if( !arg_exists(key) ) return false;
        val = std::stoi(args[key]);
        return true;
    }

    bool args_get_val(float&val,const char*tag,const float def_val=0) {
        std::string key(tag);
        val = def_val;
        if( !arg_exists(key) ) return false;
        val = std::stof(args[key]);
        return true;
    }

    bool args_get_val(VUInt3&val,const char*tag,const uint32 def_val=0) {
        std::string key(tag);
        val.x = def_val;
        val.y = def_val;
        val.z = def_val;
        if( !arg_exists(key) ) return false;

        std::stringstream ss(args[key]);
        std::string token;

        std::getline(ss, token, ',');
        val.x = static_cast<uint32_t>(std::stoul(token));

        std::getline(ss, token, ',');
        val.y = static_cast<uint32_t>(std::stoul(token));

        std::getline(ss, token, ',');
        val.z = static_cast<uint32_t>(std::stoul(token));
        return true;
    }

    bool args_get_val(Vec3&val,const char*tag,const float def_val=0) {
        std::string key(tag);
        val.x = def_val;
        val.y = def_val;
        val.z = def_val;
        if( !arg_exists(key) ) return false;

        std::stringstream ss(args[key]);
        std::string token;

        std::getline(ss, token, ',');
        val.x = std::stof(token);

        std::getline(ss, token, ',');
        val.y = std::stof(token);

        std::getline(ss, token, ',');
        val.z = std::stof(token);

        return true;
    }

    bool args_get_val(char*val,const char*tag) {
        std::string key(tag);
        val[0] = 0;
        if( !arg_exists(key) ) return false;
        strcpy(val,args[key].c_str());

        return true;
    }

protected:
    void get_line() {
        bool read_next_line = true;
        while( read_next_line ) {
            if( fgets(buffer,SUSAN_FILENAME_LENGTH,fp) != NULL ) {
                int buf_len = strlen(buffer);
                if( buf_len > 0 ) {
                    if( buffer[0] != '#' ) {
                        read_next_line = false;
                        while( buf_len > 0 && (buffer[buf_len-1] == '\n' || buffer[buf_len-1] == '\r') ) {
                            buf_len--;
                            buffer[buf_len] = 0;
                        }
                    }
                }
            }
            else {
                fprintf(stderr,"Truncated file\n");
                exit(1);
            }
        }
    }

    char*get_char_ptr(const char*tag) {
        char*rslt = buffer;
        int tag_len = strlen(tag);
        get_line();
        if( strncmp( buffer, tag, tag_len ) == 0 && buffer[tag_len] == ':' ) {
            rslt = buffer+tag_len+1;
            while( rslt[0] == ' ' || rslt[0] == '\t' ) /// optional space after the ':'
                rslt++;
        }
        else {
            fprintf(stderr,"Requested tag %s, buffer: %s\n",tag,buffer);
            exit(0);
        }
        return rslt;
    }

    void remove_commas(char*buf) {
        for(int i=0;i<SUSAN_FILENAME_LENGTH;i++) {
            if( buf[i] == 0 )
                break;
            if( buf[i] == ',' )
                buf[i] = ' ';
        }
    }

    bool arg_exists(std::string&key) {
        return args.find(key) != args.end();
    }
};

////

uint32 parse_uint32_strlist(uint32* &ptr, const char*strlist) {

    int i = 0;
    uint32 count = 1;

    while( strlist[i] > 0 ) {
        if( strlist[i] == ',' )
            count++;
        i++;
    }

    ptr = new uint32[count];

    char buffer[50];
    int j = 0, k = 0;;
    i = 0;
    while( strlist[i] > 0 ) {
        if( strlist[i] == ',' ) {
            buffer[k] = 0;
            ptr[j] = atoi(buffer);
            k = 0;
            j++;
        }
        else {
            buffer[k] = strlist[i];
            k++;
        }
        i++;
    }
    buffer[k] = 0;
    ptr[j] = atoi(buffer);

    return count;
}

uint32 parse_single_strlist(single* &ptr, const char*strlist) {

    int i = 0;
    uint32 count = 1;

    while( strlist[i] > 0 ) {
        if( strlist[i] == ',' )
            count++;
        i++;
    }

    ptr = new single[count];

    char buffer[50];
    int j = 0, k = 0;;
    i = 0;
    while( strlist[i] > 0 ) {
        if( strlist[i] == ',' ) {
            buffer[k] = 0;
            ptr[j] = atof(buffer);
            k = 0;
            j++;
        }
        else {
            buffer[k] = strlist[i];
            k++;
        }
        i++;
    }
    buffer[k] = 0;
    ptr[j] = atof(buffer);

    return count;
}

}

#endif


