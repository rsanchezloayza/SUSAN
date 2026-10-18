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

#ifndef REFERENCE_H
#define REFERENCE_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "datatypes.h"
#include "io.h"
#include "mrc.h"
#include "memory.h"

typedef struct {
    char map [ SUSAN_FILENAME_LENGTH ];
    char mask[ SUSAN_FILENAME_LENGTH ];
    char h1  [ SUSAN_FILENAME_LENGTH ];
    char h2  [ SUSAN_FILENAME_LENGTH ];
} Reference;

class References {

public:
    int num_refs;

protected:
    Reference  *refs;

public:
    References(const char*filename) {
        refs = NULL;
        IO::TxtParser parser(filename,"refstxt");
        parser.get_value(num_refs,"num_ref");

        refs = new Reference[num_refs];

        char base_dir[ SUSAN_FILENAME_LENGTH ];
        IO::get_dirname(base_dir,filename);

        for(int n=0;n<num_refs;n++) {
            parser.get_str( refs[n].map , "map"  );
            parser.get_str( refs[n].mask, "mask" );
            parser.get_str( refs[n].h1  , "h1"   );
            parser.get_str( refs[n].h2  , "h2"   );
            resolve(refs[n].map ,base_dir,filename);
            resolve(refs[n].mask,base_dir,filename);
            resolve(refs[n].h1  ,base_dir,filename);
            resolve(refs[n].h2  ,base_dir,filename);
        }

    }

    ~References() {
        free_array(refs);
    }

protected:
    void resolve(char*path,const char*base_dir,const char*filename) {
        if( path[0] == 0 )
            return;
        if( !IO::resolve_path(path,base_dir) ) {
            fprintf(stderr,"%s: file %s not found (as given, nor relative to %s).\n",filename,path,base_dir);
            exit(1);
        }
    }

public:

    Reference& operator[](int ix) {
        if( ix >= num_refs ) {
            fprintf(stderr,"Trying to access reference %d from %d.\n",ix,num_refs);
            exit(1);
        }
        return refs[ix];
    }

    Reference& at(int ix) {
        if( ix >= num_refs ) {
            fprintf(stderr,"Trying to access reference %d from %d.\n",ix,num_refs);
            exit(1);
        }
        return refs[ix];
    }

    void print() {
        for(int n=0;n<num_refs;n++) {
            printf("Reference %d\n",n);
            printf("  Map:  %s\n",refs[n].map );
            printf("  Mask: %s\n",refs[n].mask);
            if( refs[n].h1[0] > 0 )
                printf("  H1:   %s\n",refs[n].h1  );
            if( refs[n].h2[0] > 0 )
            printf("  H2:   %s\n",refs[n].h2  );

        }
    }

    bool check_fields(const bool consider_halves) {
        bool rslt = true;

        for(int n=0;n<num_refs;n++) {
            if( refs[n].map[0] == 0 ) {
                fprintf(stderr,"Error in reference %d: field 'map' empty.\n",n+1);
                rslt = false;
            }

            if( refs[n].mask[0] == 0 ) {
                fprintf(stderr,"Error in reference %d: field 'mask' empty.\n",n+1);
                rslt = false;
            }

            if( consider_halves ) {
                if( refs[n].h1[0] == 0 ) {
                    fprintf(stderr,"Error in reference %d: field 'h1' empty.\n",n+1);
                    rslt = false;
                }

                if( refs[n].h2[0] == 0 ) {
                    fprintf(stderr,"Error in reference %d: field 'h2' empty.\n",n+1);
                    rslt = false;
                }
            }
        }

        if( !rslt ) {
            fprintf(stderr,"Incomplete reference file.\n");
            exit(1);
        }

        return rslt;
    }

    bool check_size(const uint32 N,const bool consider_halves) {
        bool rslt = true;
        uint32 X,Y,Z;
        for(int n=0;n<num_refs;n++) {
            if( !IO::exists(refs[n].map) ) {
                fprintf(stderr,"File %s do not exist.\n",refs[n].map);
                exit(1);
            }
            Mrc::read_size(X,Y,Z,refs[n].map);
            if( X!=N || Y!=N || Z!=N  ) {
                fprintf(stderr,"Error on map %s with size %dx%dx%d, it should be %d\n",refs[n].map,X,Y,Z,N);
                rslt = false;
            }

            if( !IO::exists(refs[n].mask) ) {
                fprintf(stderr,"File %s do not exist.\n",refs[n].mask);
                exit(1);
            }
            Mrc::read_size(X,Y,Z,refs[n].mask);
            if( X!=N || Y!=N || Z!=N  ) {
                fprintf(stderr,"Error on map %s with size %dx%dx%d, it should be %d\n",refs[n].mask,X,Y,Z,N);
                rslt = false;
            }

            if( consider_halves ) {
                if( !IO::exists(refs[n].h1) ) {
                    fprintf(stderr,"File %s do not exist.\n",refs[n].h1);
                    exit(1);
                }
                Mrc::read_size(X,Y,Z,refs[n].h1);
                if( X!=N || Y!=N || Z!=N  ) {
                    fprintf(stderr,"Error on map %s with size %dx%dx%d, it should be %d\n",refs[n].h1,X,Y,Z,N);
                    rslt = false;
                }

                if( !IO::exists(refs[n].h2) ) {
                    fprintf(stderr,"File %s do not exist.\n",refs[n].h2);
                    exit(1);
                }
                Mrc::read_size(X,Y,Z,refs[n].h2);
                if( X!=N || Y!=N || Z!=N  ) {
                    fprintf(stderr,"Error on map %s with size %dx%dx%d, it should be %d\n",refs[n].h2,X,Y,Z,N);
                    rslt = false;
                }
            }
        }
        return rslt;
    }
};

#endif /// REFERENCE_H


