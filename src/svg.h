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

#ifndef SVG_H
#define SVG_H

#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <sstream>
#include "datatypes.h"
#include "lodepng.h"

std::string base64_encode(const uint8_t* p_data, size_t n_data) {
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    if (!p_data || n_data == 0)
        return std::string();

    size_t output_length = 4*((n_data+2)/3);
    std::string encoded;
    encoded.reserve(output_length);

    size_t i = 0;
    while(i<n_data) {
        uint32_t octet_a = i < n_data ? p_data[i++] : 0;
        uint32_t octet_b = i < n_data ? p_data[i++] : 0;
        uint32_t octet_c = i < n_data ? p_data[i++] : 0;

        uint32_t triple = (octet_a << 16) |
                          (octet_b << 8)  |
                          octet_c;

        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back(table[(triple >> 6)  & 0x3F]);
        encoded.push_back(table[triple & 0x3F]);
    }

    size_t mod = n_data % 3;
    if(mod>0) {
        encoded[output_length - 1] = '=';
        if(mod==1)
            encoded[output_length - 2] = '=';
    }

    return encoded;
}

void array_to_turbo_cmap(uint8_t* p_out,const float* p_in,int W,int H,float vmin,float vmax) {

    const int numel = W*H;

    float scale = 0.0f;
    if (vmax>vmin)
        scale = 1.0f/(vmax-vmin);

    for(int i=0;i<numel;++i) {

        float x = scale*(p_in[i]-vmin);
        x = fminf(fmaxf(x,0.0f),1.0f);

        float x2 =  x*x;
        float x3 = x2*x;
        float x4 = x3*x;
        float x5 = x4*x;

        // Turbo polynomial
        float r =     0.13572138f
                  +   4.61539260f*x
                  -  42.66032258f*x2
                  + 132.13108234f*x3
                  - 152.94239396f*x4
                  +  59.28637943f*x5;

        float g =     0.09140261f
                  +   2.19418839f*x
                  +   4.84296658f*x2
                  -  14.18503333f*x3
                  +   4.27729857f*x4
                  +   2.82956604f*x5;

        float b =     0.10667330f
                  +  12.64194608f*x
                  -  60.58204836f*x2
                  + 110.36276771f*x3
                  -  89.90310912f*x4
                  +  27.34824973f*x5;

        r = fminf(fmaxf(r,0.0f),1.0f);
        g = fminf(fmaxf(g,0.0f),1.0f);
        b = fminf(fmaxf(b,0.0f),1.0f);

        p_out[3*i  ] = (uint8_t)(roundf(r*255.0f));
        p_out[3*i+1] = (uint8_t)(roundf(g*255.0f));
        p_out[3*i+2] = (uint8_t)(roundf(b*255.0f));
    }
}

void array_to_cividis_cmap(uint8_t* p_out,const float* p_in,int W,int H,float vmin,float vmax) {

    const int numel = W*H;

    float scale = 0.0f;
    if (vmax>vmin)
        scale = 1.0f/(vmax-vmin);

    for(int i=0;i<numel;++i) {

        float x = scale*(p_in[i]-vmin);
        x = fminf(fmaxf(x,0.0f),1.0f);

        float x2 =  x*x;
        float x3 = x2*x;
        float x4 = x3*x;
        float x5 = x4*x;

        // Cividis polynomial
        float r =   0.00021894f
                  + 0.11378068f*x
                  + 2.08656617f*x2
                  - 4.77258489f*x3
                  + 3.61987054f*x4
                  - 1.00911555f*x5;

        float g =   0.00403110f
                  + 0.50732958f*x
                  + 1.60068260f*x2
                  - 4.35907890f*x3
                  + 3.18223883f*x4
                  - 0.90215879f*x5;

        float b =   0.34982430f
                  + 0.46254292f*x
                  - 1.62227345f*x2
                  + 2.66765961f*x3
                  - 1.73149361f*x4
                  + 0.48913915f*x5;

        r = fminf(fmaxf(r,0.0f),1.0f);
        g = fminf(fmaxf(g,0.0f),1.0f);
        b = fminf(fmaxf(b,0.0f),1.0f);

        p_out[3*i  ] = (uint8_t)(roundf(r*255.0f));
        p_out[3*i+1] = (uint8_t)(roundf(g*255.0f));
        p_out[3*i+2] = (uint8_t)(roundf(b*255.0f));
    }
}

void array_to_viridis_cmap(uint8_t* p_out,const float* p_in,int W,int H,float vmin,float vmax) {

    const int numel = W*H;

    float scale = 0.0f;
    if (vmax>vmin)
        scale = 1.0f/(vmax-vmin);

    for(int i=0;i<numel;++i) {

        float x = scale*(p_in[i]-vmin);
        x = fminf(fmaxf(x,0.0f),1.0f);

        float x2 =  x*x;
        float x3 = x2*x;
        float x4 = x3*x;
        float x5 = x4*x;

        // Viridis polynomial
        float r =   0.28026800f
                  + 0.23051900f * x
                  + 0.14351000f * x2
                  - 0.35108000f * x3
                  + 0.19157000f * x4
                  - 0.04216000f * x5;

        float g =   0.16536800f
                  + 1.02324000f * x
                  - 0.13231000f * x2
                  - 0.38131000f * x3
                  + 0.26384000f * x4
                  - 0.06545000f * x5;

        float b =   0.47623400f
                  + 0.63103000f * x
                  - 1.18146000f * x2
                  + 1.35776000f * x3
                  - 0.82564000f * x4
                  + 0.20030000f * x5;

        r = fminf(fmaxf(r,0.0f),1.0f);
        g = fminf(fmaxf(g,0.0f),1.0f);
        b = fminf(fmaxf(b,0.0f),1.0f);

        p_out[3*i  ] = (uint8_t)(roundf(r*255.0f));
        p_out[3*i+1] = (uint8_t)(roundf(g*255.0f));
        p_out[3*i+2] = (uint8_t)(roundf(b*255.0f));
    }
}

void ctf_ellipse_fit_to_svg(const char*fname,const float*p_in,int N,float vmin,float vmax,float apix,float U,float V,float ang) {

    std::vector<uint8_t> rgb(N*N*3);
    //array_to_turbo_cmap(rgb.data(),p_in,N,N,vmin,vmax);
    array_to_viridis_cmap(rgb.data(),p_in,N,N,vmin,vmax);
    //array_to_cividis_cmap(rgb.data(),p_in,N,N,vmin,vmax);

    std::vector<uint8_t> png_data;
    unsigned error = lodepng::encode(png_data,rgb,N,N,LCT_RGB);
    if(error)
        throw std::runtime_error("PNG encoding failed");

    std::string b64 = base64_encode(png_data.data(), png_data.size());

    float cx = N/2.0f+0.5f;
    float cy = N/2.0f+0.5f;

    float theta= ang*RAD2DEG;

    float umstep = 2.0f*1e4f/apix;

    std::ostringstream svg;

    svg << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\"\n";
    svg << "     width=\"" << N << "\"\n";
    svg << "     height=\"" << N << "\"\n";
    svg << "     viewBox=\"0 0 " << N << " " << N << "\">\n";

    svg << "<g transform=\"translate(0," << N
        << ") scale(1,-1)\">\n";

    // Image
    svg << "<image href=\"data:image/png;base64,"
        << b64
        << "\" width=\"" << N
        << "\" height=\"" << N
        << "\" style=\"image-rendering: pixelated; image-rendering: crisp-edges;\"/>\n";

    // Grid
    svg << "<line x1=\"0\" y1=\"" << cy
        << "\" x2=\"" << N
        << "\" y2=\"" << cy
        << "\" style=\"stroke:#333333;stroke-width:0.5\" />\n";

    svg << "<line x1=\"" << cx
        << "\" y1=\"0\" x2=\"" << cx
        << "\" y2=\"" << N
        << "\" style=\"stroke:#333333;stroke-width:0.5\" />\n";

    for(float r=umstep;r<N/2.0f;r+=umstep) {
        svg << "<circle r=\"" << r
            << "\" cx=\"" << cx
            << "\" cy=\"" << cy
            << "\" fill=\"none\" stroke=\"#333333\" stroke-width=\"0.5\" />\n";
    }

    // Defocus fitting
    svg << "<ellipse cx=\"" << cx
        << "\" cy=\"" << cy
        << "\" rx=\"" << U
        << "\" ry=\"" << V
        << "\" fill=\"none\" stroke=\"#e64343\" stroke-width=\"0.8\" "
        << "transform=\"rotate(" << theta
        << "," << cx << "," << cy << ")\" />\n";

    svg << "</g>\n";

    // Add grid markers
    for(int k=2;k<11;k+=2) {
        //<text x="5" y="30" fill="pink" stroke="blue" font-size="35">I love SVG!</text>
        float r = k*umstep/2;
        svg << "<text x=\"" << (cx+r+1)
            << "\" y=\"" << (cy+6)
            << "\" fill=\"#333333\" stroke=\"none\" font-size=\"8\">" << k
            << "µm</text>\n";
    }

    svg << "</svg>\n";

    std::ofstream file(fname, std::ios::out | std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open SVG file");

    file << svg.str();



}

class SvgCtf {

public:
    SvgCtf(const char*filename,const float in_apix) {
        fp = fopen(filename,"w");
        apix = in_apix;
        fprintf(fp,"<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" height=\"530\" width=\"800\" style=\"fill:white;stroke:black;stroke-width:2;\" font-family=\"Arial,Helvetica,sans-serif\">\n");
        has_avg = false;
        has_est = false;
        has_fit = false;
    }

    ~SvgCtf() {
        fprintf(fp,"</svg>\n");
        fclose(fp);
    }

    void create_grid(float res_min, float res_max, float N) {
        float y_val[] = {1,0.75,0.5,0.143,0};
        float x,y;

        /// Background and Grid
        fprintf(fp,"  <rect x=\"60\" y=\"40\" width=\"720\" height=\"400\" style=\"stroke-width:0\"/>\n");
        fprintf(fp,"  <g style=\"fill:" SUSAN_SVG_SHADOW_BG ";stroke-width:0\">\n");
        if( res_min > 0 ) {
            x = 720*(2*res_min/N);
            fprintf(fp,"  <rect x=\"60\" y=\"40\" width=\"%.1f\" height=\"400\"/>\n",x);
        }
        if( res_max > 0 ) {
            x = 720*(2*res_max/N);
            fprintf(fp,"  <rect x=\"%.1f\" y=\"40\" width=\"%.1f\" height=\"400\"/>\n",x+60,720-x);
        }
        fprintf(fp,"  </g>\n");
        fprintf(fp,"  <g style=\"stroke:#E6E6E6\">\n");
        for(int i=1;i<10;i++) {
            x = 72.0*i + 60;
            fprintf(fp,"    <line x1=\"%.1f\" y1=\"40\" x2=\"%.1f\" y2=\"440\"/>\n",x,x);
        }
        for(int i=1;i<5;i++) {
            y =(1-y_val[i])*400 + 40;
            fprintf(fp,"    <line x1=\"60\" y1=\"%.0f\" x2=\"780\" y2=\"%.0f\"/>\n",y,y);
        }
        fprintf(fp,"  </g>\n");
        fprintf(fp,"  <rect x=\"60\" y=\"40\" width=\"720\" height=\"400\" style=\"fill:none\"/>\n");
        
        /// XY label:
        fprintf(fp,"  <text x=\"20\" y=\"240\" dominant-baseline=\"middle\" text-anchor=\"middle\" transform=\"rotate(-90 20 240)\" style=\"fill:black;stroke:none;font-size:18px;\">Normalized Amplitude</text>\n");
        fprintf(fp,"  <text x=\"420\" y=\"482\" dominant-baseline=\"middle\" text-anchor=\"middle\" style=\"fill:black;stroke:none;font-size:18px;\">Resolution (Å)</text>\n");

        /// Y ticks:
        fprintf(fp,"  <g>\n");
        for(int i=0;i<5;i++) {
            y =(1-y_val[i])*400 + 40;
            fprintf(fp,"    <text x=\"55\" y=\"%.2f\" dominant-baseline=\"middle\" text-anchor=\"end\" style=\"fill:black;stroke:none;font-size:12px;\">%.2f</text>\n",y,y_val[i]);
        }
        fprintf(fp,"  </g>\n");

        /// X ticks:
        fprintf(fp,"  <g>\n");
        for(int i=1;i<=10;i++) {
            float x_apix = 20*apix/i;
            x = 72*i + 60;
            fprintf(fp,"    <text x=\"%.2f\" y=\"445\" dominant-baseline=\"middle\" text-anchor=\"end\" transform=\"rotate(-45 %.2f 445)\" style=\"fill:black;stroke:none;font-size:12px;\">%.2f</text>\n",x,x,x_apix);
        }
        fprintf(fp,"  </g>\n");
    }

    void create_title(const int n_proj,const float def) {
        fprintf(fp,"  <text x=\"400\" y=\"20\" dominant-baseline=\"middle\" text-anchor=\"middle\" style=\"fill:black;stroke:none;font-weight:bold;font-size:20px\">Average Defocus for projection %d: %.2fÅ</text>\n",n_proj,def);
    }

    void add_avg(const float*ptr,const float M) {
        add_signal(ptr,M,SUSAN_SVG_FG_A);
        has_avg = true;
    }

    void add_fit(const float*ptr,const float M,bool do_square=false) {
        add_signal(ptr,M,SUSAN_SVG_FG_B);
        has_fit = true;
    }

    void add_est(const float*ptr,const float M,bool as_signal=false) {
        if( as_signal )
            add_signal(ptr,M,SUSAN_SVG_FG_C);
        else {
            fprintf(fp,"  <g style=\"stroke:" SUSAN_SVG_FG_C ";fill:none\">\n");
            fprintf(fp,"    <polyline points=\"60,40");
            float x,y,prev_x=60,prev_y=40;
            for(int i=0;i<M;i++) {
                if( ptr[i] > 0 ) {
                    x = i;
                    x = 60 + 720*(x/(M-1));
                    y = (1-ptr[i])*400 + 40;
                    fprintf(fp," %.2f,%.2f",(prev_x+x)/2,prev_y);
                    fprintf(fp," %.2f,%.2f",(prev_x+x)/2,y);
                    prev_x = x;
                    prev_y = y;
                }
            }
            fprintf(fp," 780,%.2f",y);
            fprintf(fp,"\" />\n");
            fprintf(fp,"  </g>\n");
        }
        has_est = true;
    }

    void create_legend() {
        fprintf(fp,"  <g>\n");
        fprintf(fp,"    <rect x=\"60\" y=\"495\" width=\"720\" height=\"25\"/>\n");
        if( has_avg ) {
            create_legend_entry("Radial Average",SUSAN_SVG_FG_A,60+15);
        }
        if( has_fit ) {
            create_legend_entry("Estimated CTF",SUSAN_SVG_FG_B,60+15+220);
        }
        if( has_est ) {
            create_legend_entry("Phase matching coefficient",SUSAN_SVG_FG_C,60+15+220+220);
        }
        fprintf(fp,"  </g>\n");
    }

protected:
    FILE*fp;
    float apix;

    bool has_avg;
    bool has_fit;
    bool has_est;

    void add_signal(const float*ptr,const float M,const char*color,bool do_square=false) {
        fprintf(fp,"  <g style=\"stroke:%s;fill:none\">\n",color);
        fprintf(fp,"    <polyline points=\"");
        for(int i=0;i<M;i++) {
            if(i>0)
                fprintf(fp," ");
            float x = i;
            x = 60 + 720*(x/(M-1));
            float y = (1-ptr[i])*400 + 40;
            if(do_square)
                y = (1-(ptr[i]*ptr[i]))*400 + 40;
            fprintf(fp,"%.2f,%.2f",x,y);
        }
        fprintf(fp,"\" />\n");
        fprintf(fp,"  </g>\n");
    }

    void create_legend_entry(const char*entry,const char*color,const int offset) {
        fprintf(fp,"    <line x1=\"%d\" y1=\"507.5\" x2=\"%d\" y2=\"507.5\" style=\"stroke:%s\"/>\n",offset,offset+40,color);
        fprintf(fp,"    <text x=\"%d\" y= \"507.5\" dominant-baseline=\"middle\" text-anchor=\"start\" style=\"fill:black;stroke:none;font-size:16px;\">%s</text>\n",offset+50,entry);
    }

};

class SvgFsc{

public:
    SvgFsc(const char*filename,const float in_apix) {
        fp = fopen(filename,"w");
        apix = in_apix;
        fprintf(fp,"<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" height=\"530\" width=\"800\" style=\"fill:white;stroke:black;stroke-width:2;\" font-family=\"Arial,Helvetica,sans-serif\">\n");
        has_unm = false;
        has_msk = false;
        has_rnd = false;
    }

    ~SvgFsc() {
        fprintf(fp,"</svg>\n");
        fclose(fp);
    }

    void create_grid(float fpix_rand, float res_max, float threshold, float N) {

        float x,y;
        /// Background and Grid
        fprintf(fp,"  <rect x=\"40\" y=\"40\" width=\"740\" height=\"400\" style=\"stroke-width:0\"/>\n");
        fprintf(fp,"  <g style=\"fill:" SUSAN_SVG_SHADOW_BG ";stroke-width:0\">\n");
        x = 740*(fpix_rand/(N/2+1));
        fprintf(fp,"  <rect x=\"40\" y=\"40\" width=\"%.1f\" height=\"400\"/>\n",x);
        x = 740*(2*apix/res_max);
        fprintf(fp,"  <rect x=\"%.1f\" y=\"40\" width=\"%.1f\" height=\"400\"/>\n",x+40,740-x);
        fprintf(fp,"  </g>\n");
        fprintf(fp,"  <g style=\"stroke:#E6E6E6\">\n");
        for(int i=1;i<15;i++) {
            x = 740.0*i/15.0 + 40;
            fprintf(fp,"    <line x1=\"%.1f\" y1=\"40\" x2=\"%.1f\" y2=\"440\"/>\n",x,x);
        }
        y =(1-threshold)*400 + 40;
        fprintf(fp,"    <line x1=\"40\" y1=\"%.0f\" x2=\"780\" y2=\"%.0f\"/>\n",y,y);
        fprintf(fp,"  </g>\n");
        fprintf(fp,"  <rect x=\"40\" y=\"40\" width=\"740\" height=\"400\" style=\"fill:none\"/>\n");

        /// X label:
        fprintf(fp,"  <text x=\"420\" y=\"482\" dominant-baseline=\"middle\" text-anchor=\"middle\" style=\"fill:black;stroke:none;font-size:18px;\">Resolution (Å)</text>\n");

        /// Y ticks:
        fprintf(fp,"  <g>\n");
        y = 400 + 40;
        fprintf(fp,"    <text x=\"35\" y=\"%.2f\" dominant-baseline=\"middle\" text-anchor=\"end\" style=\"fill:black;stroke:none;font-size:12px;\">0.00</text>\n",y);
        y =(1.0-threshold)*400 + 40;
        fprintf(fp,"    <text x=\"35\" y=\"%.2f\" dominant-baseline=\"middle\" text-anchor=\"end\" style=\"fill:black;stroke:none;font-size:12px;\">%.2f</text>\n",y,threshold);
        y = 40;
        fprintf(fp,"    <text x=\"35\" y=\"%.2f\" dominant-baseline=\"middle\" text-anchor=\"end\" style=\"fill:black;stroke:none;font-size:12px;\">1.00</text>\n",y);
        fprintf(fp,"  </g>\n");

        /// X ticks:
        fprintf(fp,"  <g>\n");
        for(int i=1;i<=15;i++) {
            float x_apix = 2.0*15.0*apix/i;
            x = 740.0*i/15.0 + 40;
            fprintf(fp,"    <text x=\"%.2f\" y=\"445\" dominant-baseline=\"middle\" text-anchor=\"end\" transform=\"rotate(-45 %.2f 445)\" style=\"fill:black;stroke:none;font-size:12px;\">%.2f</text>\n",x,x,x_apix);
        }
        fprintf(fp,"  </g>\n");
    }

    void create_title(const int n_ref,const float res) {
        fprintf(fp,"  <text x=\"400\" y=\"20\" dominant-baseline=\"middle\" text-anchor=\"middle\" style=\"fill:black;stroke:none;font-weight:bold;font-size:20px\">FSC Class %d: %.3fÅ</text>\n",n_ref,res);
    }

    void add_unmask(const float*ptr,const float M) {
        add_signal(ptr,M,SUSAN_SVG_FG_A);
        has_unm = true;
    }

    void add_masked(const float*ptr,const float M) {
        add_signal(ptr,M,SUSAN_SVG_FG_B);
        has_msk = true;
    }

    void add_rndmzd(const float*ptr,const float M) {
        add_signal(ptr,M,SUSAN_SVG_FG_C);
        has_rnd = true;
    }

    void create_legend() {
        fprintf(fp,"  <g>\n");
        fprintf(fp,"    <rect x=\"60\" y=\"495\" width=\"720\" height=\"25\"/>\n");
        if( has_unm ) {
            create_legend_entry("Unmasked",SUSAN_SVG_FG_A,75);
        }
        if( has_msk ) {
            create_legend_entry("Masked",SUSAN_SVG_FG_B,330);
        }
        if( has_rnd ) {
            create_legend_entry("Phase randomized",SUSAN_SVG_FG_C,580);
        }
        fprintf(fp,"  </g>\n");
    }

protected:
    FILE*fp;
    float apix;

    bool has_unm;
    bool has_msk;
    bool has_rnd;

    void add_signal(const float*ptr,const float M,const char*color) {
        fprintf(fp,"  <g style=\"stroke:%s;fill:none\">\n",color);
        fprintf(fp,"    <polyline points=\"");
        for(int i=0;i<M;i++) {
            if(i>0)
                fprintf(fp," ");
            float x = i;
            x = 40 + 740*(x/(M-1));
            float y = (1-ptr[i])*400 + 40;
            fprintf(fp,"%.2f,%.2f",x,y);
        }
        fprintf(fp,"\" />\n");
        fprintf(fp,"  </g>\n");
    }

    void create_legend_entry(const char*entry,const char*color,const int offset) {
        fprintf(fp,"    <line x1=\"%d\" y1=\"507.5\" x2=\"%d\" y2=\"507.5\" style=\"stroke:%s\"/>\n",offset,offset+40,color);
        fprintf(fp,"    <text x=\"%d\" y= \"507.5\" dominant-baseline=\"middle\" text-anchor=\"start\" style=\"fill:black;stroke:none;font-size:16px;\">%s</text>\n",offset+50,entry);
    }

};

#endif 

