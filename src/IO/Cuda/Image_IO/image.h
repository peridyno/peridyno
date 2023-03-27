/*
 * @file image.h 
 * @brief Image class, support basic operations on image data.
 * @author FeiZhu
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_IMAGE_IMAGE_H_
#define PHYSIKA_CORE_IMAGE_IMAGE_H_

namespace dyno{

template <typename T, int Dim> class Range;

class Image
{
public:
    enum DataFormat{
        RGB,
        RGBA
    };
public:
    Image(); //construct an empty image
    //construct image with given data, data are copied internally
    Image(unsigned int width, unsigned int height, DataFormat data_format, const unsigned char *raw_data);
    Image(const Image &image);  //copy constructor, data is copied 
    Image& operator= (const Image &image);
    ~Image();

    //basic getters&&setters
    unsigned int width() const;
    unsigned int height() const;
    DataFormat dataFormat() const;
    const unsigned char* rawData() const;
    unsigned char* rawData();
    void setRawData(unsigned int width, unsigned int height, DataFormat data_format, const unsigned char *raw_data);    //data is copied

    //image operations
    void flipHorizontally();
    void flipVertically();
    Image mirrorImage() const;
    Image upsideDownImage() const;
protected:
    void allocMemory();
    unsigned int pixelSize() const;
protected:
    unsigned int width_;
    unsigned int height_;
    DataFormat data_format_;
    unsigned char *raw_data_;
};

}  //end of namespace dyno

#endif //PHYSIKA_CORE_IMAGE_IMAGE_H_
