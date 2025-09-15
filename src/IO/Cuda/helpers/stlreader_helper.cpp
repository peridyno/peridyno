#include "tinyobj_helper.h"

#include "ImageLoader.h"

#include "stl_reader/stl_reader.h"

namespace dyno
{
	bool loadTextureMeshFromStl(std::shared_ptr<TextureMesh> texMesh, const FilePath& fullname, bool useToCenter)
	{
		//tinyobj::attrib_t attrib;
		//std::vector<tinyobj::shape_t> shapes;
		//std::vector<tinyobj::material_t> materials;
		//std::string err, warn;

		//auto name = fullname;
		//auto root = name.path().parent_path();

		//bool result = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
		//	name.string().c_str(), root.string().c_str());

		//if (!err.empty()) {
		//	std::cerr << err << std::endl;
		//	return false;
		//}

		//if (!warn.empty()) {
		//	std::cerr << warn << std::endl;
		//}

		//std::vector<dyno::Vec3f> vertices;
		//std::vector<dyno::Vec3f> normals;
		//std::vector<dyno::Vec2f> texCoords;
		//std::vector<dyno::uint> shapeIds;

		//std::vector<dyno::Vec3i> pIndex;
		//std::vector<dyno::Vec3i> nIndex;
		//std::vector<dyno::Vec3i> tIndex;

		//for (int i = 0; i < attrib.vertices.size(); i += 3) {
		//	vertices.push_back({ attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2] });
		//}

		//for (int i = 0; i < attrib.normals.size(); i += 3) {
		//	normals.push_back({ attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2] });
		//}

		//for (int i = 0; i < attrib.texcoords.size(); i += 2) {
		//	texCoords.push_back({ attrib.texcoords[i], attrib.texcoords[i + 1] });
		//}

		//shapeIds.resize(vertices.size());
		//texMesh->shapeIds().reset();
		//// load texture...
		//dyno::CArray2D<dyno::Vec4f> texture(1, 1);
		//texture[0, 0] = dyno::Vec4f(1);

		//auto& tMats = texMesh->materials();
		//tMats.resize(materials.size());

		//uint mId = 0;
		//for (const auto& mtl : materials) {
		//	tMats[mId] = std::make_shared<Material>();
		//	//tMats[mId]->ambient = { mtl.ambient[0], mtl.ambient[1], mtl.ambient[2] };
		//	//tMats[mId]->diffuse = { mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2] };
		//	//tMats[mId]->specular = { mtl.specular[0], mtl.specular[1], mtl.specular[2] };
		//	//tMats[mId]->roughness = 1.0f - mtl.shininess;
		//	tMats[mId]->baseColor = { mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2] };

		//	std::shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>();

		//	if (!mtl.diffuse_texname.empty())
		//	{
		//		auto tex_path = (root / mtl.diffuse_texname).string();

		//		if (loader->loadImage(tex_path.c_str(), texture))
		//		{
		//			tMats[mId]->texColor.assign(texture);
		//		}
		//	}
		//	if (!mtl.bump_texname.empty())
		//	{
		//		auto tex_path = (root / mtl.bump_texname).string();

		//		if (loader->loadImage(tex_path.c_str(), texture))
		//		{
		//			tMats[mId]->texBump.assign(texture);
		//			auto texOpt = mtl.bump_texopt;
		//			tMats[mId]->bumpScale = texOpt.bump_multiplier;
		//		}
		//	}

		//	mId++;
		//}


		//auto& tShapes = texMesh->shapes();
		//tShapes.resize(shapes.size());

		//uint sId = 0;

		//for (const tinyobj::shape_t& shape : shapes) {
		//	// only load triangle mesh...
		//	const auto& mesh = shape.mesh;
		//	tShapes[sId] = std::make_shared<Shape>();
		//	std::vector<TopologyModule::Triangle> vertexIndex;
		//	std::vector<TopologyModule::Triangle> normalIndex;
		//	std::vector<TopologyModule::Triangle> texCoordIndex;

		//	if (mesh.material_ids.size() > 0 && mesh.material_ids[0] >= 0)
		//	{
		//		tShapes[sId]->material = tMats[mesh.material_ids[0]];
		//	}

		//	Vec3f lo = Vec3f(REAL_MAX);
		//	Vec3f hi = Vec3f(-REAL_MAX);
		//	for (int i = 0; i < mesh.indices.size(); i += 3) {
		//		auto idx0 = mesh.indices[i];
		//		auto idx1 = mesh.indices[i + 1];
		//		auto idx2 = mesh.indices[i + 2];

		//		vertexIndex.push_back({ idx0.vertex_index, idx1.vertex_index, idx2.vertex_index });
		//		normalIndex.push_back({ idx0.normal_index, idx1.normal_index, idx2.normal_index });
		//		texCoordIndex.push_back({ idx0.texcoord_index, idx1.texcoord_index, idx2.texcoord_index });

		//		lo = lo.minimum(vertices[idx0.vertex_index]);
		//		lo = lo.minimum(vertices[idx1.vertex_index]);
		//		lo = lo.minimum(vertices[idx2.vertex_index]);

		//		hi = hi.maximum(vertices[idx0.vertex_index]);
		//		hi = hi.maximum(vertices[idx1.vertex_index]);
		//		hi = hi.maximum(vertices[idx2.vertex_index]);

		//		shapeIds[idx0.vertex_index] = sId;
		//		shapeIds[idx1.vertex_index] = sId;
		//		shapeIds[idx2.vertex_index] = sId;

		//	}
		//	tShapes[sId]->vertexIndex.assign(vertexIndex);
		//	tShapes[sId]->normalIndex.assign(normalIndex);
		//	tShapes[sId]->texCoordIndex.assign(texCoordIndex);

		//	auto shapeCenter = (lo + hi) / 2;
		//	tShapes[sId]->boundingBox = TAlignedBox3D<Real>(lo, hi);
		//	tShapes[sId]->boundingTransform = Transform3f(shapeCenter, Quat1f().toMatrix3x3());

		//	
		//	//Move to center
		//	if (useToCenter) 
		//	{
		//		std::vector<int> indicator(vertices.size(), 0);
		//		for (int i = 0; i < mesh.indices.size(); i += 3)
		//		{
		//			auto idx0 = mesh.indices[i];
		//			auto idx1 = mesh.indices[i + 1];
		//			auto idx2 = mesh.indices[i + 2];

		//			if (indicator[idx0.vertex_index] == 0)
		//			{
		//				vertices[idx0.vertex_index] -= shapeCenter;
		//				indicator[idx0.vertex_index] = 1;
		//			}
		//			if (indicator[idx1.vertex_index] == 0)
		//			{
		//				vertices[idx1.vertex_index] -= shapeCenter;
		//				indicator[idx1.vertex_index] = 1;
		//			}
		//			if (indicator[idx2.vertex_index] == 0)
		//			{
		//				vertices[idx2.vertex_index] -= shapeCenter;
		//				indicator[idx2.vertex_index] = 1;
		//			}
		//		}
		//	}
		//	

		//	vertexIndex.clear();
		//	normalIndex.clear();
		//	texCoordIndex.clear();

		//	sId++;
		//}

		//texMesh->vertices().assign(vertices);
		//texMesh->normals().assign(normals);
		//texMesh->texCoords().assign(texCoords);
		//texMesh->shapeIds().assign(shapeIds);

		////A hack: for an obj file with one shape
		//if (shapes.size() == 1)
		//{
		//	texMesh->shapeIds().resize(vertices.size());
		//	texMesh->shapeIds().reset();
		//}

		//vertices.clear();
		//normals.clear();
		//texCoords.clear();

		return true;
	}


	bool loadStl(std::vector<Vec3f>& points, std::vector<TopologyModule::Triangle>& triangles, std::string filename, bool append)
	{
		if (!append)
		{
			points.clear();
			triangles.clear();
		}
		int offset = append ? points.size() : 0;


		std::vector<float> coords;
		std::vector<float> normals;
		std::vector<size_t> tris;
		std::vector<size_t> solids;

		try {
			stl_reader::ReadStlFile(filename.c_str(), coords, normals, tris, solids);
			const size_t numTris = tris.size() / 3;
		}
		catch (const std::exception& e) {
			std::cerr << "Error reading STL file: " << e.what() << std::endl;
		}

		std::cout << "************************    Loading  STL    ************************  " << std::endl << std::endl;
		std::cout << "                        " << "    coords size = " << coords.size() << std::endl << std::endl;
		std::cout << "                        " << "    normal size =" << normals.size()  << std::endl << std::endl;
		std::cout << "                        " << "    tris size = " << tris.size()  << std::endl << std::endl;
		std::cout << "                        " << "    solids size =" << solids.size() << std::endl << std::endl;

		if (solids.size() == 0) { return false; }

		for (int i = 0; i < coords.size() / 3; i++)
		{
			points.push_back(Vec3f(coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]));
		}
		std::cout << "************************    Loading : f    ************************  " << std::endl << std::endl;

		//std::cout << "                        " << "    Triangle " << i << " size =" << myshape[i].mesh.indices.size() / 3 << std::endl << std::endl;

		for (int s = 0; s < tris.size() / 3; s++)
		{
			//std::cout << myshape[i].mesh.indices[s].vertex_index <<"  " << std::endl;

			triangles.push_back(TopologyModule::Triangle(tris[3 * s] + offset, tris[3 * s + 1] + offset, tris[3 * s + 2] + offset));
		}
		
		std::cout << "************************    Loading completed    **********************" << std::endl << std::endl;
		return true;

	}

}