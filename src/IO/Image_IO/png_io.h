/*
 * @file png_io.h 
 * @Brief load/save png file
 * @author Fei Zhu, WeiChen
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_IO_IMAGE_IO_PNG_IO_H_
#define PHYSIKA_IO_IMAGE_IO_PNG_IO_H_

#include <string>
#include "image.h"

namespace dyno{

class PngIO
{
public:
    PngIO(){}
    ~PngIO(){}

    /// warning: this function only read color data(IDAT chunk) from png file
    ///	(i.e. it ignores all other data chunks ,such as ancillary chunks)
    /// since only IDAT chunk makes sense for our texture.Thus if you load data from a png file and
    ///  resave image data to another png file,the file size will be smaller than the origin one.
    static bool load(const std::string &filename, Image * image);  //data_format = RGBA
    static bool load(const std::string &filename, Image * image, Image::DataFormat data_format);

    /* save image data to file, the image data is in row order
     * return true if succeed, otherwise return false
     */
    static bool save(const std::string &filename, const Image *image);
protected:

};

} //end of namespace dyno

#endif //PHYSIKA_IO_IMAGE_IO_PNG_IO_H_
