/*
 * @file ppm_io.cpp 
 * @Brief load/save ppm file
 * @author Wei Chen
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "Image_IO/image_io.h"
#include "Image_IO/ppm_io.h"
#include "Algorithm.h"
#include "File_Utilities/file_content_utilities.h"
#include "File_Utilities/file_path_utilities.h"
using std::string;
using std::getline;
using std::fstream;
using std::stringstream;
using std::ifstream;
using std::ofstream;

namespace dyno{

bool PPMIO::load(const string &filename,Image *image )
{
    return PPMIO::load(filename, image, Image::DataFormat::RGBA);
}

bool PPMIO::load(const std::string &filename, Image * image, Image::DataFormat data_format)
{
    if(ImageIO::checkFileNameAndImage(filename,string(".ppm"),image) == false)
        return false;

    fstream fp;
    fp.open(filename.c_str(),std::ios::in|std::ios::binary);
    if(!fp.is_open())
    {
        std::cerr<<"Couldn't opern .ppm file:"<<filename<<std::endl;
        return false;
    }
    string file_head;
    while(getline(fp, file_head))
    {
        file_head = FileUtilities::removeWhitespaces(file_head,1);
        if(file_head.at(0) == '#')
            continue;
        else
            break;
    }

    if(file_head.substr(0,2) != "P6")
    {
        std::cerr<<"file is not raw RGB ppm\n";
        return false;
    }
    int file_para[3];
    int para_num = 0;
    while(para_num<3 && getline(fp, file_head)  )
    {
        file_head = FileUtilities::removeWhitespaces(file_head,1);
        if(file_head.at(0) == '#')
            continue;
        else
        {
            if(file_head.find('#') != file_head.npos)
                file_head.substr(0,file_head.find('#'));
            file_head = FileUtilities::removeWhitespaces(file_head,1);
            stringstream strstream(file_head);
            while(strstream>>file_para[para_num])  // important sentence
            { 
                para_num++;
            }
        }
    }
    unsigned char * image_data = new unsigned char [ 3*file_para[0]*file_para[1] ];
    if(image_data == NULL)
    {
        std::cerr<<"error in allocating memory!";
        return false;
    }
    // read image RGB data
    int data_size = fp.read( (char *)image_data,sizeof(unsigned char)*3*file_para[0]*file_para[1]).gcount();
    if( data_size < 3*file_para[0]*file_para[1])
    {
        std::cerr<<"error in reading image RGB data";
    }
    if(data_format == Image::RGB)
    {
        image->setRawData(file_para[0], file_para[1], Image::RGB, image_data);
    }
    else
    {
        unsigned char * image_data_with_alpha = new unsigned char [ 4*file_para[0]*file_para[1] ];
        if(image_data_with_alpha == NULL)
        {
            std::cerr<<"error in allocating memory!";
            return false;
        }
        unsigned int num_pixel = file_para[0] * file_para[1];
        for(unsigned int i=0 ; i<num_pixel; i++)
        {
            image_data_with_alpha[4*i]   = image_data[3*i];
            image_data_with_alpha[4*i+1] = image_data[3*i+1];
            image_data_with_alpha[4*i+2] = image_data[3*i+2];
            image_data_with_alpha[4*i+3] = 255;
        }
        image->setRawData(file_para[0], file_para[1], Image::RGBA, image_data_with_alpha);
        delete [] image_data_with_alpha;
    }
    delete [] image_data;
    fp.close();
    return true;
}

bool PPMIO::save(const string &filename, const Image *image)
{
    if(ImageIO::checkFileNameAndImage(filename,string(".ppm"),image) == false)
        return false;

    fstream fp;
    fp.open(filename.c_str(),std::ios::out|std::ios::binary);
    if(!fp.is_open())
    {
        std::cerr<<"error in opening file!"<<std::endl;
        return false;
    }
    char char_0A=11;
    fp<<"P6"<<char_0A;
    fp<<image->width()<<char_0A;
    fp<<image->height()<<char_0A;
    fp<<"255"<<char_0A;
    
    unsigned int            num_pixel = image->width()*image->height();
    const unsigned  char *  row_data  = image->rawData();
    if(image->dataFormat() == Image::RGB)
    {
        for(unsigned int i=0; i<num_pixel; i++ )
        {
            fp<<row_data[3*i];
            fp<<row_data[3*i+1];
            fp<<row_data[3*i+2];
        }
    }
    else
    {
        for(unsigned int i=0; i<num_pixel; i++ )
        {
            fp<<row_data[4*i];
            fp<<row_data[4*i+1];
            fp<<row_data[4*i+2];
        }
    }
    
    fp.close();
    return true;
}

} //end of namespace dyno
