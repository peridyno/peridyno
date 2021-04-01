/*
 * @file image_io.cpp 
 * @Brief image_io class, it is used to import/save image files such as bmp etc.
 * @author Sheng Yang, Fei Zhu
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include "File_Utilities/file_path_utilities.h"
#include "Image_IO/image_io.h"
#include "Image_IO/ppm_io.h"
using std::string;

namespace dyno{


bool ImageIO::load(const string & filename, Image* image)
{
    return ImageIO::load(filename, image, Image::RGBA);
}

bool ImageIO::load(const string &filename, Image * image, Image::DataFormat data_format)
{
    string suffix = FileUtilities::fileExtension(filename);
	if(suffix==string(".ppm"))
        return PPMIO::load(filename, image, data_format);
    else
    {
        std::cerr<<"Unknown image file format!\n";
        return false;
    }
}

bool ImageIO::save(const string &filename, const Image * image)
{
    string suffix = FileUtilities::fileExtension(filename);
    if(suffix==string(".ppm"))
            return PPMIO::save(filename, image);
    else
    {
        std::cerr<<"Unknown image file format specified!\n";
        return false;
    }
}

bool ImageIO::checkFileNameAndImage(const std::string &filename, const std::string &expected_extension, const Image *image)
{
    std::string file_extension = FileUtilities::fileExtension(filename);
    if(file_extension.size() == 0)
    {
        std::cerr<<"No file extension found for the image file:"<<filename<<std::endl;
        return false;
    }
    if(file_extension != expected_extension)
    {
        std::cerr<<"Unknown file format:"<<file_extension<<std::endl;
        return false;
    }
    if(image == NULL)
    {
        std::cerr<<"NULL image passed to ImageIO"<<std::endl;
        return false;
    }
    return true;
}

} //end of namespace dyno
