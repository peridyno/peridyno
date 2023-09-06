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
	this->stateTriangleSet()->promoteOuput();
}

void ObjMeshNode::resetStates()
{
	auto fullname = this->varFileName()->getValue();
	auto root = fullname.parent_path();

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err, warn;

	bool result = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
		fullname.string().c_str(), root.string().c_str());

	if (!err.empty()) {
		std::cerr << err << std::endl;
		return;
	}

	if (!warn.empty()) {
		std::cerr << warn << std::endl;
	}

	std::vector<dyno::Vec3f> vertices;
	std::vector<dyno::Vec3f> normals;
	std::vector<dyno::Vec2f> texCoords;

	std::vector<dyno::Vec3i> pIndex;
	std::vector<dyno::Vec3i> nIndex;
	std::vector<dyno::Vec3i> tIndex;

	for (int i = 0; i < attrib.vertices.size(); i += 3) {
		vertices.push_back({ attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2] });
	}

	for (int i = 0; i < attrib.normals.size(); i += 3) {
		normals.push_back({ attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2] });
	}

	for (int i = 0; i < attrib.texcoords.size(); i += 2) {
		texCoords.push_back({ attrib.texcoords[i], attrib.texcoords[i + 1] });
	}

	this->stateShapes()->resize(shapes.size());
	auto& statShapes = this->stateShapes()->getData();

	uint sId = 0;
	for (const tinyobj::shape_t& shape : shapes) {
		// only load triangle mesh...
		const auto& mesh = shape.mesh;
		statShapes[sId] = std::make_shared<gl::Shape>();
		std::vector<dyno::TopologyModule::Triangle> vertexIndex;
		std::vector<dyno::TopologyModule::Triangle> normalIndex;
		std::vector<dyno::TopologyModule::Triangle> texCoordIndex;
		for (int i = 0; i < mesh.indices.size(); i += 3) {
			auto idx0 = mesh.indices[i];
			auto idx1 = mesh.indices[i + 1];
			auto idx2 = mesh.indices[i + 2];

			pIndex.push_back({ idx0.vertex_index, idx1.vertex_index, idx2.vertex_index });
			nIndex.push_back({ idx0.normal_index, idx1.normal_index, idx2.normal_index });
			tIndex.push_back({ idx0.texcoord_index, idx1.texcoord_index, idx2.texcoord_index });

			vertexIndex.push_back({ idx0.vertex_index, idx1.vertex_index, idx2.vertex_index });
			normalIndex.push_back({ idx0.normal_index, idx1.normal_index, idx2.normal_index });
			texCoordIndex.push_back({ idx0.texcoord_index, idx1.texcoord_index, idx2.texcoord_index });
		}
		statShapes[sId]->vertexIndex.assign(vertexIndex);
		statShapes[sId]->normalIndex.assign(normalIndex);
		statShapes[sId]->texCoordIndex.assign(texCoordIndex);

		vertexIndex.clear();
		normalIndex.clear();
		texCoordIndex.clear();

		sId++;
	}

	// load texture...
	dyno::CArray2D<dyno::Vec4f> texture(1, 1);
	texture[0, 0] = dyno::Vec4f(1);

	this->stateMaterials()->resize(materials.size());
	auto& sMats = this->stateMaterials()->getData();

	uint mId = 0;
	for (const auto& mtl : materials) {
		sMats[mId] = std::make_shared<gl::Material>();
		sMats[mId]->ambient = { mtl.ambient[0], mtl.ambient[1], mtl.ambient[2] };
		sMats[mId]->diffuse = { mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2] };
		sMats[mId]->specular = { mtl.specular[0], mtl.specular[1], mtl.specular[2] };
		sMats[mId]->roughness = 1.0f - mtl.shininess;
		if (!mtl.diffuse_texname.empty())
		{
			auto tex_path = (root / mtl.diffuse_texname).string();

			// load textures
			loadImage(tex_path.c_str(), texture);
			sMats[mId]->texColor.assign(texture);
		}
		if (!mtl.bump_texname.empty())
		{
			auto tex_path = (root / mtl.bump_texname).string();

			// load textures
			loadImage(tex_path.c_str(), texture);
			sMats[mId]->texBump.assign(texture);
		}

		mId++;
	}

	if (this->stateTriangleSet()->isEmpty())
	{
		this->stateTriangleSet()->allocate();
	}

	auto ts = this->stateTriangleSet()->getDataPtr();

	ts->setPoints(vertices);
	ts->setTriangles(pIndex);
	ts->update();

	auto t = this->varLocation()->getValue();
	auto q = this->computeQuaternion();
	auto s = this->varScale()->getValue();

	ts->scale(s);
	ts->rotate(q);
	ts->translate(t);

	this->stateVertex()->assign(vertices);
	this->stateNormal()->assign(normals);
	this->stateTexCoord()->assign(texCoords);
	this->stateNormalIndex()->assign(nIndex);
	this->stateTexCoordIndex()->assign(tIndex);
	this->stateColorTexture()->assign(texture);

	this->update();
}
