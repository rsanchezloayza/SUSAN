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

#include "datatypes.h"
#include "math_cpu.h"

class ProgressReporter {

protected:
    char  *caption;
    char  *buffer;
    single total;
    single current;
    int    l_buffer;

    Math::Timing timer;

    void set_caption(single total_progress,bool show_buffering) {
        int offset=0;
        offset = sprintf(buffer,"\r%s: %6.2f%% ",caption,100.0*total_progress/total);

        if( total_progress > 0 ) {
            int days,hours,mins,secs;
            timer.get_etc(days,hours,mins,secs,total_progress,total);

            if( days > 0 ) {
                offset = offset + sprintf(buffer+offset,"(ETA: %dd %02d:%02d:%02d)",days,hours,mins,secs);
            }
            else {
                offset = offset + sprintf(buffer+offset,"(ETA: %02d:%02d:%02d)",hours,mins,secs);
            }
        }
        else {
            offset = offset + sprintf(buffer+offset,"(ETA: TBD)");
        }

        if( show_buffering ) {
            offset = offset + sprintf(buffer+offset," [Buffering]");
        }

        int i;
        for(i=offset;i<l_buffer-1;i++)
            buffer[i] = ' ';
        buffer[i] = 0;

        fprintf(stdout,"%s", buffer);
        fflush(stdout);
    }

    void set_finished() {
        int offset=0;
        offset = sprintf(buffer,"\r%s: 100.00%%",caption);

        int i;
        for(i=offset;i<l_buffer-1;i++)
            buffer[i] = ' ';
        buffer[i] = 0;

        fprintf(stdout,"%s\n", buffer);
        fflush(stdout);
    }
public:
    ProgressReporter(const char*in_caption,const uint32 in_total) {
        int l_cap = strlen(in_caption);
        caption   = new char[l_cap+1];
        l_buffer  = l_cap + strlen(": PPP.PP% (ETA: XXXd XX:XX:XX) [Buffering] ") + 2;
        buffer    = new char[l_buffer];
        total     = (single)in_total;
        current   = 0;
        strcpy(caption,in_caption);
        memset((void*)buffer,0,sizeof(char)*(l_buffer));
    }

    ~ProgressReporter() {
        delete [] caption;
        delete [] buffer;
    }

    void set_total(const uint32 in_total) {
        total = (single)in_total;
    }

    void start() {
        timer.tic();
        set_caption(0,true);
    }

    void update(const uint32 total_progress,bool show_buffering) {
        set_caption(total_progress,show_buffering);
    }

    void finish() {
        set_finished();
    }
};

