// 图形加载
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
// 必须放最前面
#include <glad/glad.h>  // Initialize with gladLoadGL()
#include <GLFW/glfw3.h>

typedef struct Picture {
    GLuint texture = 0;
    int image_height = 0;
    int image_width = 0;
    Picture(const char *filename) {
        bool ret = LoadTextureFromFile(filename, &texture, &image_width,
                                       &image_height);
        // IM_ASSERT(ret);
    }
    // 加载纹理
    bool LoadTextureFromFile(const char *filename, GLuint *out_texture,
                             int *out_width, int *out_height);
    void *GetTexture() { return (void *)(intptr_t)texture; }
} Picture;

// Simple helper function to load an image into a OpenGL texture with common
// settings
bool Picture::LoadTextureFromFile(const char *filename, GLuint *out_texture,
                                  int *out_width, int *out_height) {
    // Load from file
    int image_width = 0;
    int image_height = 0;
    unsigned char *image_data =
        stbi_load(filename, &image_width, &image_height, NULL, 4);
    if (image_data == NULL) return false;

    // Create a OpenGL texture identifier
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Upload pixels into texture
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, image_data);
    stbi_image_free(image_data);

    *out_texture = image_texture;
    *out_width = image_width;
    *out_height = image_height;

    return true;
}