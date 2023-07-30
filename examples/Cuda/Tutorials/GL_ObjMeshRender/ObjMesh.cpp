#include "ObjMesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <iostream>
#include <filesystem>

bool loadImage(const char* path, dyno::CArray2D<dyno::Vec4f>& img) 
{
    int x, y, comp;
    stbi_set_flip_vertically_on_load(true);
    float* data = stbi_loadf(path, &x, &y, &comp, STBI_default);

    if (data) {
        img.resize(x, y);
        for (int x0 = 0; x0 < x; x0++)
        {
            for (int y0 = 0; y0 < y; y0++)
            {
                int idx = (y0 * x + x0) * comp;

                for (int c0 = 0; c0 < comp; c0++) {
                    img(x0, y0)[c0] = data[idx + c0];
                }
            }
        }
        return true;
    }

    STBI_FREE(data);

    return false;
}

ObjMeshNode::ObjMeshNode() {
}

bool ObjMeshNode::load(const std::string& path) {
    auto root = std::filesystem::path(path).parent_path();

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err, warn;

    bool result = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, 
        path.c_str(), root.string().c_str());

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!warn.empty()) {
        std::cerr << warn << std::endl;
    }

    std::vector<dyno::Vec3f> vertices;
    std::vector<dyno::Vec3f> normals;
    std::vector<dyno::Vec2f> texCoords;
    std::vector<dyno::Vec3i> triangles;

    for (int i = 0; i < attrib.vertices.size(); i += 3) {
        vertices.push_back({ attrib.vertices[i], attrib.vertices[i+1], attrib.vertices[i+2] });
    }

    for (int i = 0; i < attrib.normals.size(); i += 3) {
        normals.push_back({ attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2] });
    }

    for (int i = 0; i < attrib.texcoords.size(); i += 2) {       
        texCoords.push_back({ attrib.texcoords[i], attrib.texcoords[i + 1]});
    }

    for (const tinyobj::shape_t& shape: shapes) {
        // only load triangle mesh...
        const auto& mesh = shape.mesh;
        for (const auto& idx : mesh.indices) {
            triangles.push_back({ idx.vertex_index, idx.normal_index, idx.texcoord_index });
        }
    }

    // load texture...
    dyno::CArray2D<dyno::Vec4f> texture(1, 1);
    texture[0, 0] = dyno::Vec4f(1, 0, 0, 1);

    for (const auto& mtl : materials) {
        if (!mtl.diffuse_texname.empty())
        {
            auto tex_path = (root / mtl.diffuse_texname).string();
            std::cout << tex_path << std::endl;

            // load textures
            loadImage(tex_path.c_str(), texture);
            break;
        }
    }

    this->out_Position.assign(vertices);
    this->out_Normal.assign(normals);
    this->out_TexCoord.assign(texCoords);
    this->out_Index.assign(triangles);
    this->out_TexColor.assign(texture);
    this->update();

    return result;
}
