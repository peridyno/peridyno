/*
 * @file png_io.h 
 * @Brief load/save ppm file
 * @author WeiChen
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_IO_IMAGE_IO_PPM_IO_H_
#define PHYSIKA_IO_IMAGE_IO_PPM_IO_H_

#include <string>
//#include "image_io.h"
#include "image.h"

namespace dyno{

class PPMIO
{
public:
    PPMIO(){}
    ~PPMIO(){}

    static bool load(const std::string &filename, Image * image);  //data_format = RGBA
    static bool load(const std::string &filename, Image * image, Image::DataFormat data_format);

    /* save image data to file, the image data is in row order
     * return true if succeed, otherwise return false
     */
    static bool save(const std::string &filename, const Image *image);
protected:

};

} //end of namespace dyno

#endif //PHYSIKA_IO_IMAGE_IO_PPM_IO_H_
