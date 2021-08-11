#ifndef _PICTURE_H
#define _PICTURE_H

// 必须放最前面
#include <glad/glad.h>  // Initialize with gladLoadGL()
#include <assert.h>
// 图形加载
// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"
namespace dyno {
	typedef struct Picture {
		GLuint texture = 0;
		int image_height = 0;
		int image_width = 0;
		Picture(const char *filename) {
			bool ret = LoadTextureFromFile(filename, &texture, &image_width,
				&image_height);
			assert(ret);
		}
		// 加载纹理
		bool LoadTextureFromFile(const char *filename, GLuint *out_texture,
			int *out_width, int *out_height);
		void *GetTexture() { return (void *)(intptr_t)texture; }
	} Picture;
}
#endif // !_PICTURE_H