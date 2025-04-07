#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "Node.h"

namespace dyno {

    class ImageLoader 
    {
    public:

        ImageLoader() {}
        ~ImageLoader() {}
        static bool loadImage(const char* path, CArray2D<Vec4f>& img);
    };

} 

#endif // IMAGE_LOADER_H