/*
 * @file png_io.cpp 
 * @Brief load/save png file
 * @author Wei Chen, Fei Zhu
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
#include <sstream>
#include "Image_IO/image_io.h"
#include "Image_IO/png_io.h"
#include "LodePNG/lodepng.h"
#include "Utility.h"

using std::string;

namespace dyno{


bool PngIO::load(const string &filename,Image *image )
{
    return PngIO::load(filename, image, Image::RGBA);
}

bool PngIO::load(const std::string &filename, Image * image, Image::DataFormat data_format)
{
    if(ImageIO::checkFileNameAndImage(filename,string(".png"),image) == false)
        return false;

    unsigned int width, height;
    unsigned int error;
    std::vector<unsigned char> image_vec;

    if(data_format == Image::RGBA) //RGBA format
    {
        error = lodepng::decode(image_vec, width, height, filename);   //decode png file to image
    }
    else //RGB format
    {
        error = lodepng::decode(image_vec, width, height, filename, LCT_RGB);
    }
    std::stringstream adaptor;
    adaptor<<error;
    string error_str;
    adaptor>>error_str;
    string error_message = "decoder error "+error_str+string(": ")+lodepng_error_text(error);
    if(error!=0)
    {
        std::cerr<<error_message<<std::endl;
        return false;
    }
    
    unsigned char * image_data;
    if(data_format == Image::RGBA) //RGBA format
    {
        image_data= new unsigned char[width*height*4];  //allocate memory
    }
    else
    {
        image_data= new unsigned char[width*height*3];
    }
    if(image_data == NULL)
    {
        std::cerr<<"error: can't allocate memory!"<<std::endl;
        return false;
    }
    if(data_format == Image::RGBA)   //RGBA format
    {
        for(unsigned int i=0; i<image_vec.size(); i=i+4) //loop for perPixel
        {
            image_data[i] = image_vec[i];        // red   color
            image_data[i+1] = image_vec[i+1];    // green color
            image_data[i+2] = image_vec[i+2];    // blue  color
            image_data[i+3] = image_vec[i+3];    // alpha value
        }
    }
    else
    {
        for(unsigned int i=0; i<image_vec.size(); i=i+3) //loop for perPixel
        {
            image_data[i] = image_vec[i];        // red   color
            image_data[i+1] = image_vec[i+1];    // green color
            image_data[i+2] = image_vec[i+2];    // blue  color
        }
    }
    image->setRawData(width, height,data_format, image_data);
    delete [] image_data;
    return true;
}

bool PngIO::save(const string &filename, const Image *image)
{
    if(ImageIO::checkFileNameAndImage(filename,string(".png"),image) == false)
        return false;

    unsigned int error;
    if(image->dataFormat() == Image::RGBA)
    {
        error = lodepng::encode(filename, image->rawData(), image->width(), image->height());   //encode the image_data to file
    }
    else
    {
        error = lodepng::encode(filename, image->rawData(), image->width(), image->height(), LCT_RGB);
    }
    std::stringstream adaptor;
    adaptor<<error;
    string error_str;
    adaptor>>error_str;
    string error_message = "decoder error "+error_str+string(": ")+lodepng_error_text(error);   //define the error message 
    if(error!=0)
    {
        std::cerr<<error_message<<std::endl;
        return false;
    }                                     
    return true;
}

} //end of namespace dyno
