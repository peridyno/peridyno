/*
 * @file image_io.h 
 * @Brief image_io class, it is used to import/save image files such as bmp etc.
 * @author Fei Zhu
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_IO_IMAGE_IO_IMAGE_IO_H_
#define PHYSIKA_IO_IMAGE_IO_IMAGE_IO_H_

#include <string>
#include "image.h"

namespace dyno{
class ImageIO
{
public:
    ImageIO(){}
    ~ImageIO(){}
 
    static bool load(const std::string &filename, Image * image);  //data_format = RGBA
    static bool load(const std::string &filename, Image * image, Image::DataFormat data_format);

    /* save image data to file, the image data is in row order
     * return true if succeed, otherwise return false
     */
    static bool save(const std::string &filename, const Image* image);

    //check if the input filename and image is valid, called in load and save of specific IO classes
    //return true if check succeeds
    static bool checkFileNameAndImage(const std::string &filename, const std::string &expected_extension,
                                      const Image *image);

protected:
};

} //end of namespace dyno

#endif //PHYSIKA_IO_IMAGE_IO_IMAGE_IO_H_
