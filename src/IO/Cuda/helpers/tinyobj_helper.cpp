#include "tinyobj_helper.h"

#include "Array/Array.h"
#include "ImageLoader.h"
#include "Module/TopologyModule.h"
#include "Topology/TextureMesh.h"
#include "Vector/Vector3D.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

namespace dyno
{
	bool loadTextureMeshFromObj(std::shared_ptr<TextureMesh> texMesh, const FilePath& fullname, bool dotransform)
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err, warn;

		auto name = fullname;
		auto root = name.path().parent_path();

		bool result = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
			name.string().c_str(), root.string().c_str());

		if (!err.empty()) {
			std::cerr << err << std::endl;
			return false;
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

		// load texture...
		dyno::CArray2D<dyno::Vec4f> texture(1, 1);
		texture[0, 0] = dyno::Vec4f(1);

		auto& tMats = texMesh->materials();
		tMats.resize(materials.size());

		uint mId = 0;
		for (const auto& mtl : materials) {
			tMats[mId] = std::make_shared<Material>();
			//tMats[mId]->ambient = { mtl.ambient[0], mtl.ambient[1], mtl.ambient[2] };
			//tMats[mId]->diffuse = { mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2] };
			//tMats[mId]->specular = { mtl.specular[0], mtl.specular[1], mtl.specular[2] };
			//tMats[mId]->roughness = 1.0f - mtl.shininess;
			tMats[mId]->baseColor = { mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2] };

			std::shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>();

			if (!mtl.diffuse_texname.empty())
			{
				auto tex_path = (root / mtl.diffuse_texname).string();

				if (loader->loadImage(tex_path.c_str(), texture))
				{
					tMats[mId]->texColor.assign(texture);
				}
			}
			if (!mtl.bump_texname.empty())
			{
				auto tex_path = (root / mtl.bump_texname).string();

				if (loader->loadImage(tex_path.c_str(), texture))
				{
					tMats[mId]->texBump.assign(texture);
					auto texOpt = mtl.bump_texopt;
					tMats[mId]->bumpScale = texOpt.bump_multiplier;
				}
			}

			mId++;
		}


		auto& tShapes = texMesh->shapes();
		tShapes.resize(shapes.size());

		uint sId = 0;
		for (const tinyobj::shape_t& shape : shapes) {
			// only load triangle mesh...
			const auto& mesh = shape.mesh;
			tShapes[sId] = std::make_shared<Shape>();
			std::vector<TopologyModule::Triangle> vertexIndex;
			std::vector<TopologyModule::Triangle> normalIndex;
			std::vector<TopologyModule::Triangle> texCoordIndex;

			if (mesh.material_ids.size() > 0 && mesh.material_ids[0] >= 0)
			{
				tShapes[sId]->material = tMats[mesh.material_ids[0]];
			}

			Vec3f lo = Vec3f(REAL_MAX);
			Vec3f hi = Vec3f(-REAL_MAX);
			for (int i = 0; i < mesh.indices.size(); i += 3) {
				auto idx0 = mesh.indices[i];
				auto idx1 = mesh.indices[i + 1];
				auto idx2 = mesh.indices[i + 2];

				vertexIndex.push_back({ idx0.vertex_index, idx1.vertex_index, idx2.vertex_index });
				normalIndex.push_back({ idx0.normal_index, idx1.normal_index, idx2.normal_index });
				texCoordIndex.push_back({ idx0.texcoord_index, idx1.texcoord_index, idx2.texcoord_index });

				lo = lo.minimum(vertices[idx0.vertex_index]);
				lo = lo.minimum(vertices[idx1.vertex_index]);
				lo = lo.minimum(vertices[idx2.vertex_index]);

				hi = hi.maximum(vertices[idx0.vertex_index]);
				hi = hi.maximum(vertices[idx1.vertex_index]);
				hi = hi.maximum(vertices[idx2.vertex_index]);
			}
			tShapes[sId]->vertexIndex.assign(vertexIndex);
			tShapes[sId]->normalIndex.assign(normalIndex);
			tShapes[sId]->texCoordIndex.assign(texCoordIndex);

			auto shapeCenter = (lo + hi) / 2;
			if(!dotransform)
				shapeCenter = Vec3f(0);
			tShapes[sId]->boundingBox = TAlignedBox3D<Real>(lo, hi);
			tShapes[sId]->boundingTransform = Transform3f(shapeCenter, Quat1f().toMatrix3x3());

			//Move to center
			std::vector<int> indicator(vertices.size(), 0);
			for (int i = 0; i < mesh.indices.size(); i += 3)
			{
				auto idx0 = mesh.indices[i];
				auto idx1 = mesh.indices[i + 1];
				auto idx2 = mesh.indices[i + 2];

				if (indicator[idx0.vertex_index] == 0)
				{
					vertices[idx0.vertex_index] -= shapeCenter;
					indicator[idx0.vertex_index] = 1;
				}
				if (indicator[idx1.vertex_index] == 0)
				{
					vertices[idx1.vertex_index] -= shapeCenter;
					indicator[idx1.vertex_index] = 1;
				}
				if (indicator[idx2.vertex_index] == 0)
				{
					vertices[idx2.vertex_index] -= shapeCenter;
					indicator[idx2.vertex_index] = 1;
				}
			}

			vertexIndex.clear();
			normalIndex.clear();
			texCoordIndex.clear();

			sId++;
		}

		texMesh->vertices().assign(vertices);
		texMesh->normals().assign(normals);
		texMesh->texCoords().assign(texCoords);

		vertices.clear();
		normals.clear();
		texCoords.clear();

		return true;
	}


    bool manualParseSceneConfig(const std::string& xmlPath, std::vector<SceneObject>& sceneObjects, std::vector<Asset>& assets) {
        tinyxml2::XMLDocument doc;
        tinyxml2::XMLError loadResult = doc.LoadFile(xmlPath.c_str());
        if (loadResult != tinyxml2::XML_SUCCESS) {
            std::cerr << "Error: Could not load XML file :" << xmlPath << std::endl;
            return false;
        }

        tinyxml2::XMLElement* sceneElement = doc.FirstChildElement("Scene");
        if (!sceneElement) {
            std::cerr << "Error: Could not find <Scene> element in the XML file." << std::endl;
            return false;
        }

        std::map<std::string, int> assetIdToIndexMap;

        // parse Assets
        tinyxml2::XMLElement* assetsElement = sceneElement->FirstChildElement("Assets");
        if (assetsElement) {
            int currentIndex = 0;
            for (tinyxml2::XMLElement* assetElement = assetsElement->FirstChildElement("Asset"); assetElement != nullptr; assetElement = assetElement->NextSiblingElement("Asset")) {
                Asset currentAsset;

                const char* assetId = assetElement->Attribute("id");
                if (assetId) {
                    currentAsset.name = assetId;
                    assetIdToIndexMap[currentAsset.name] = currentIndex;
                }

                tinyxml2::XMLElement* modelElement = assetElement->FirstChildElement("Model");

                if (modelElement) {
                    currentAsset.modelPath = modelElement->GetText();
                }

                tinyxml2::XMLElement* matElement = assetElement->FirstChildElement("Mat");
                if (matElement) {
                    currentAsset.matPath = matElement->GetText();
                }

                assets.push_back(currentAsset);
                currentIndex++;
            }
        }

        // load object
        for (tinyxml2::XMLElement* objectElement = assetsElement->NextSiblingElement("Object"); objectElement != nullptr; objectElement = objectElement->NextSiblingElement("Object"))
        {
            SceneObject currentObject;
            currentObject.name = objectElement->Attribute("name");
            const char* assetIdStr = objectElement->Attribute("asset_id");
            if (assetIdStr && assetIdToIndexMap.count(assetIdStr)) {
                currentObject.asset_id = assetIdToIndexMap[assetIdStr];
            }
            else {
                std::cerr << "Warning: Object '" << currentObject.name << "' has an invalid or missing asset_id." << std::endl;
                currentObject.asset_id = -1; 
            }

            tinyxml2::XMLElement* physics = objectElement->FirstChildElement("Physics");
            if (physics) {
                tinyxml2::XMLElement* density = physics->FirstChildElement("Density");
                if (density) {
                    density->QueryFloatText(&currentObject.density);
                }
                tinyxml2::XMLElement* initialVel = physics->FirstChildElement("InitialVelocity");
                if (initialVel) {
                    tinyxml2::XMLElement* linear = initialVel->FirstChildElement("Linear");
                    if (linear) {
                        linear->QueryFloatAttribute("x", &currentObject.linearVelocity.x);
                        linear->QueryFloatAttribute("y", &currentObject.linearVelocity.y);
                        linear->QueryFloatAttribute("z", &currentObject.linearVelocity.z);
                    }
                    tinyxml2::XMLElement* angular = initialVel->FirstChildElement("Angular");
                    if (angular) {
                        angular->QueryFloatAttribute("x", &currentObject.angularVelocity.x);
                        angular->QueryFloatAttribute("y", &currentObject.angularVelocity.y);
                        angular->QueryFloatAttribute("z", &currentObject.angularVelocity.z);
                    }
                }
            }

            tinyxml2::XMLElement* transform = objectElement->FirstChildElement("Transform");
            if (transform) {
                tinyxml2::XMLElement* pos = transform->FirstChildElement("Position");
                if (pos) {
                    pos->QueryFloatAttribute("x", &currentObject.position.x);
                    pos->QueryFloatAttribute("y", &currentObject.position.y);
                    pos->QueryFloatAttribute("z", &currentObject.position.z);
                }
                tinyxml2::XMLElement* orient = transform->FirstChildElement("Orientation");
                if (orient) {
                    orient->QueryFloatAttribute("pitch", &currentObject.orientation.x); // Using x for pitch
                    orient->QueryFloatAttribute("yaw", &currentObject.orientation.y);   // Using y for yaw
                    orient->QueryFloatAttribute("roll", &currentObject.orientation.z);  // Using z for roll
                }
                tinyxml2::XMLElement* scale = transform->FirstChildElement("Scale");
                if (scale) {
                    scale->QueryFloatAttribute("x", &currentObject.scale.x);
                    scale->QueryFloatAttribute("y", &currentObject.scale.y);
                    scale->QueryFloatAttribute("z", &currentObject.scale.z);
                }
            }
            sceneObjects.push_back(currentObject);
        }
    }

    void computeMassProperties(const std::vector<Vec3f>& vertices, const std::vector<TopologyModule::Triangle>& faces, Real& out_volume, Vec3f& out_center, Mat3f& out_inertia)
    {
        Real total_volume = 0.0;
        Vec3f com_accumulator;
        Mat3f inertia_accumulator;

        for (const auto& face : faces)
        {
            const Vec3f& p1 = vertices[face.x];
            const Vec3f& p2 = vertices[face.y];
            const Vec3f& p3 = vertices[face.z];

            Real volume = p1.dot(p2.cross(p3)) / 6.0;
            total_volume += volume;

            Vec3f tetra_com = (p1 + p2 + p3) / 4.0;

            com_accumulator = com_accumulator + (tetra_com * volume);

            Real scale = volume;

            Real x1 = p1.x, y1 = p1.y, z1 = p1.z;
            Real x2 = p2.x, y2 = p2.y, z2 = p2.z;
            Real x3 = p3.x, y3 = p3.y, z3 = p3.z;

            Real Ixx_expr = (y1 * y1 + y1 * y2 + y2 * y2 + y1 * y3 + y2 * y3 + y3 * y3) + (z1 * z1 + z1 * z2 + z2 * z2 + z1 * z3 + z2 * z3 + z3 * z3);
            Real Iyy_expr = (x1 * x1 + x1 * x2 + x2 * x2 + x1 * x3 + x2 * x3 + x3 * x3) + (z1 * z1 + z1 * z2 + z2 * z2 + z1 * z3 + z2 * z3 + z3 * z3);
            Real Izz_expr = (x1 * x1 + x1 * x2 + x2 * x2 + x1 * x3 + x2 * x3 + x3 * x3) + (y1 * y1 + y1 * y2 + y2 * y2 + y1 * y3 + y2 * y3 + y3 * y3);
            Real Ixy_expr = (2 * x1 * y1 + x2 * y1 + x3 * y1 + x1 * y2 + 2 * x2 * y2 + x3 * y2 + x1 * y3 + x2 * y3 + 2 * x3 * y3);
            Real Ixz_expr = (2 * x1 * z1 + x2 * z1 + x3 * z1 + x1 * z2 + 2 * x2 * z2 + x3 * z2 + x1 * z3 + x2 * z3 + 2 * x3 * z3);
            Real Iyz_expr = (2 * y1 * z1 + y2 * z1 + y3 * z1 + y1 * z2 + 2 * y2 * z2 + y3 * z2 + y1 * z3 + y2 * z3 + 2 * y3 * z3);

            Mat3f tetra_inertia;

            tetra_inertia(0, 0) = scale / 10.0 * Ixx_expr;
            tetra_inertia(1, 1) = scale / 10.0 * Iyy_expr;
            tetra_inertia(2, 2) = scale / 10.0 * Izz_expr;
            tetra_inertia(0, 1) = -(scale / 20.0 * Ixy_expr);
            tetra_inertia(0, 2) = -(scale / 20.0 * Ixz_expr);
            tetra_inertia(1, 2) = -(scale / 20.0 * Iyz_expr);
            tetra_inertia(1, 0) = tetra_inertia(0, 1);
            tetra_inertia(2, 0) = tetra_inertia(0, 2);
            tetra_inertia(2, 1) = tetra_inertia(1, 2);

            inertia_accumulator = inertia_accumulator + tetra_inertia;
        }

        out_center = com_accumulator / total_volume;

        Real cx = out_center.x, cy = out_center.y, cz = out_center.z;
        Mat3f parallel_axis_term;
        parallel_axis_term(0, 0) = total_volume * (cy * cy + cz * cz);
        parallel_axis_term(1, 1) = total_volume * (cx * cx + cz * cz);
        parallel_axis_term(2, 2) = total_volume * (cx * cx + cy * cy);
        parallel_axis_term(0, 1) = parallel_axis_term(1, 0) = -total_volume * cx * cy;
        parallel_axis_term(0, 2) = parallel_axis_term(2, 0) = -total_volume * cx * cz;
        parallel_axis_term(1, 2) = parallel_axis_term(2, 1) = -total_volume * cy * cz;

        out_inertia = inertia_accumulator - parallel_axis_term;
        out_volume = total_volume;
    }


    bool loadObjects(std::shared_ptr<TextureMesh> texMesh, std::vector<Asset>& assets, std::vector<SceneObject>& sceneObjects, bool doTransform)
    {
        std::vector<dyno::Vec3f> allVertices;
        std::vector<dyno::Vec3f> allNormals;
        std::vector<dyno::Vec2f> allTexCoords;

        auto& tMats = texMesh->materials();
        auto& tShapes = texMesh->shapes();

        bool loadedSomething = false;

        for (auto& asset : assets)
        {
            const std::string& filename = getAssetPath() + asset.modelPath;

            fs::path filePath(filename);

            if (!fs::exists(filePath) || !fs::is_regular_file(filePath) || filePath.extension() != ".obj") {
                std::cerr << "Error: Invalid file path or not an .obj file: " << filename << std::endl;
                continue;
            }

            std::cout << "Loading object from: " << filename << std::endl;

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string warn, err;

            // The directory containing the .obj file is used as the base path for materials
            std::string root = filePath.parent_path().string();

            bool result = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                filename.c_str(), root.c_str());

            if (!err.empty()) {
                std::cerr << "Error loading " << filename << ": " << err << std::endl;
                continue;
            }
            if (!warn.empty()) {
                std::cerr << "Warning from " << filename << ": " << warn << std::endl;
            }
            if (!result) {
                continue;
            }

            loadedSomething = true;

            const uint32_t vertexOffset = static_cast<uint32_t>(allVertices.size());
            const uint32_t normalOffset = static_cast<uint32_t>(allNormals.size());
            const uint32_t texCoordOffset = static_cast<uint32_t>(allTexCoords.size());
            const uint32_t materialOffset = static_cast<uint32_t>(tMats.size());

            std::vector<dyno::Vec3f> currentVertices;
            for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
                currentVertices.push_back({ attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2] });
            }

            for (size_t i = 0; i < attrib.normals.size(); i += 3) {
                allNormals.push_back({ attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2] });
            }

            for (size_t i = 0; i < attrib.texcoords.size(); i += 2) {
                allTexCoords.push_back({ attrib.texcoords[i], attrib.texcoords[i + 1] });
            }

            for (const auto& mtl : materials) {
                auto newMat = std::make_shared<Material>();
                newMat->baseColor = { mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2] };

                std::shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>();
                dyno::CArray2D<dyno::Vec4f> texture;

                if (!mtl.diffuse_texname.empty()) {
                    auto tex_path = (fs::path(root) / mtl.diffuse_texname).string();
                    if (loader->loadImage(tex_path.c_str(), texture)) {
                        newMat->texColor.assign(texture);
                    }
                }
                if (!mtl.bump_texname.empty()) {
                    auto tex_path = (fs::path(root) / mtl.bump_texname).string();
                    if (loader->loadImage(tex_path.c_str(), texture)) {
                        newMat->texBump.assign(texture);
                        newMat->bumpScale = mtl.bump_texopt.bump_multiplier;
                    }
                }
                tMats.push_back(newMat);
            }

            for (const tinyobj::shape_t& shape : shapes) {
                const auto& mesh = shape.mesh;
                auto newShape = std::make_shared<Shape>();

                std::vector<TopologyModule::Triangle> localVertexIndex;
                std::vector<TopologyModule::Triangle> vertexIndex;
                std::vector<TopologyModule::Triangle> normalIndex;
                std::vector<TopologyModule::Triangle> texCoordIndex;

                using IndexType = int;

                if (!mesh.material_ids.empty() && mesh.material_ids[0] >= 0) {
                    newShape->material = tMats[materialOffset + mesh.material_ids[0]];
                }

                Vec3f lo(REAL_MAX);
                Vec3f hi(-REAL_MAX);

                for (const auto& index : mesh.indices) {
                    lo = lo.minimum(currentVertices[index.vertex_index]);
                    hi = hi.maximum(currentVertices[index.vertex_index]);
                }

                for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                    auto idx0 = mesh.indices[i];
                    auto idx1 = mesh.indices[i + 1];
                    auto idx2 = mesh.indices[i + 2];

                    localVertexIndex.push_back({
                        static_cast<IndexType>(idx0.vertex_index),
                        static_cast<IndexType>(idx1.vertex_index),
                        static_cast<IndexType>(idx2.vertex_index) });

                    vertexIndex.push_back({
                        static_cast<IndexType>(vertexOffset + idx0.vertex_index),
                        static_cast<IndexType>(vertexOffset + idx1.vertex_index),
                        static_cast<IndexType>(vertexOffset + idx2.vertex_index) });

                    if (idx0.normal_index >= 0) {
                        normalIndex.push_back({
                            static_cast<IndexType>(normalOffset + idx0.normal_index),
                            static_cast<IndexType>(normalOffset + idx1.normal_index),
                            static_cast<IndexType>(normalOffset + idx2.normal_index) });
                    }

                    if (idx0.texcoord_index >= 0) {
                        texCoordIndex.push_back({
                            static_cast<IndexType>(texCoordOffset + idx0.texcoord_index),
                            static_cast<IndexType>(texCoordOffset + idx1.texcoord_index),
                            static_cast<IndexType>(texCoordOffset + idx2.texcoord_index) });
                    }
                }

                computeMassProperties(currentVertices, localVertexIndex, asset.volume, asset.baryCenter, asset.inertialMatrix);

                auto shapeCenter = asset.baryCenter;
                std::cout << asset.baryCenter << std::endl;
                if (!doTransform) {
                    shapeCenter = Vec3f(0.0f);
                }

                if (doTransform) {
                    for (auto& v : currentVertices) {
                        v -= shapeCenter;
                    }
                    lo -= shapeCenter;
                    hi -= shapeCenter;
                }

                newShape->boundingBox = TAlignedBox3D<Real>(lo, hi);
                newShape->boundingTransform = Transform3f(shapeCenter, Quat1f().toMatrix3x3());


                newShape->vertexIndex.assign(vertexIndex);
                newShape->normalIndex.assign(normalIndex);
                newShape->texCoordIndex.assign(texCoordIndex);

                tShapes.push_back(newShape);
            }

            allVertices.insert(allVertices.end(), currentVertices.begin(), currentVertices.end());
        }

        if (loadedSomething) {
            texMesh->vertices().assign(allVertices);
            texMesh->normals().assign(allNormals);
            texMesh->texCoords().assign(allTexCoords);
        }

        return loadedSomething;
    }

	
}