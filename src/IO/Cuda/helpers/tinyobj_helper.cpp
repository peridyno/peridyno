#include "tinyobj_helper.h"

#include "Array/Array.h"
#include "ImageLoader.h"
#include "Module/TopologyModule.h"
#include "Topology/TextureMesh.h"
#include "Vector/Vector3D.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>
#include <iostream>
#include <map>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

namespace dyno
{
    // Define fs alias to simplify filesystem path handling.
    namespace fs = std::filesystem;

	namespace
	{
		std::string toLowerCopy(const char* text)
		{
			if (text == nullptr)
			{
				return "";
			}

			std::string value(text);
			std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
				return static_cast<char>(std::tolower(c));
			});
			return value;
		}

		void queryVec3Attributes(tinyxml2::XMLElement* element, Vec3f& value)
		{
			if (element == nullptr)
			{
				return;
			}

			element->QueryFloatAttribute("x", &value.x);
			element->QueryFloatAttribute("y", &value.y);
			element->QueryFloatAttribute("z", &value.z);
		}

		SceneMotionType parseSceneMotionType(const char* text)
		{
			const std::string value = toLowerCopy(text);
			if (value == "static")
			{
				return SceneMotionType::Static;
			}
			if (value == "kinematic")
			{
				return SceneMotionType::Kinematic;
			}
			return SceneMotionType::Dynamic;
		}

		SceneCollisionProxyType parseSceneCollisionProxyType(const char* text)
		{
			const std::string value = toLowerCopy(text);
			if (value == "box" || value == "bbox" || value == "obb")
			{
				return SceneCollisionProxyType::Box;
			}
			if (value == "mat" || value == "medial" || value == "medialaxis" || value == "medial_axis")
			{
				return SceneCollisionProxyType::Mat;
			}
			return SceneCollisionProxyType::Auto;
		}

		SceneCollisionProxyType parseSceneCollisionProxyElement(tinyxml2::XMLElement* element)
		{
			if (element == nullptr)
			{
				return SceneCollisionProxyType::Auto;
			}

			const char* proxyText = element->Attribute("value");
			if (proxyText == nullptr)
			{
				proxyText = element->Attribute("type");
			}
			if (proxyText == nullptr)
			{
				proxyText = element->GetText();
			}

			return parseSceneCollisionProxyType(proxyText);
		}

		SceneJointType parseSceneJointType(const char* text)
		{
			const std::string value = toLowerCopy(text);
			if (value == "ballandsocket" || value == "ball_and_socket" || value == "ball-socket")
			{
				return SceneJointType::BallAndSocket;
			}
			if (value == "slider")
			{
				return SceneJointType::Slider;
			}
			if (value == "hinge")
			{
				return SceneJointType::Hinge;
			}
			if (value == "fixed")
			{
				return SceneJointType::Fixed;
			}
			if (value == "point")
			{
				return SceneJointType::Point;
			}
			return SceneJointType::Unknown;
		}

		std::string readObjectRef(tinyxml2::XMLElement* element)
		{
			if (element == nullptr)
			{
				return "";
			}

			const char* objectName = element->Attribute("object");
			if (objectName != nullptr)
			{
				return objectName;
			}

			const char* text = element->GetText();
			return text != nullptr ? std::string(text) : std::string();
		}

		bool isWorldRef(const std::string& name)
		{
			const std::string value = toLowerCopy(name.c_str());
			return value.empty() || value == "world";
		}
	}

    bool loadTextureMeshFromObj(std::shared_ptr<TextureMesh> texMesh, const FilePath& fullname, bool useToCenter)
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
        std::vector<dyno::uint> shapeIds;

        for (int i = 0; i < attrib.vertices.size(); i += 3) {
            vertices.push_back({ attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2] });
        }

        for (int i = 0; i < attrib.normals.size(); i += 3) {
            normals.push_back({ attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2] });
        }

        for (int i = 0; i < attrib.texcoords.size(); i += 2) {
            texCoords.push_back({ attrib.texcoords[i], attrib.texcoords[i + 1] });
        }

        // 初始化 shapeIds
        shapeIds.resize(vertices.size(), 0);

        texMesh->geometry()->shapeIds().resize(vertices.size());
        texMesh->geometry()->shapeIds().reset();

        dyno::CArray2D<dyno::Vec4f> texture(1, 1);
        texture(0, 0) = dyno::Vec4f(1);

        std::vector<std::shared_ptr<Material>> tMats;
        tMats.resize(materials.size());

        uint mId = 0;
        for (const auto& mtl : materials) {

            auto findMat = MaterialManager::getMaterialPtr(mtl.name);
            if (findMat)
            {
                std::cout << "The material already exists: " << mtl.name << std::endl;
                tMats[mId] = findMat;
                mId++;
                continue;
            }

            auto newMat = std::make_shared<Material>();
            tMats[mId] = newMat;

            MaterialManager::createMaterialLoaderModule(newMat, mtl.name);

            tMats[mId]->baseColor = Color(mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2]);
            tMats[mId]->roughness = 1.0f - mtl.shininess;

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

        Vec3f globalCenter(0.0f);
        if (useToCenter && !vertices.empty())
        {
            Vec3f globalLo = Vec3f(REAL_MAX);
            Vec3f globalHi = Vec3f(-REAL_MAX);
            for (const auto& v : vertices) {
                globalLo = globalLo.minimum(v);
                globalHi = globalHi.maximum(v);
            }
            globalCenter = (globalLo + globalHi) / 2.0f;

            for (auto& v : vertices) {
                v -= globalCenter;
            }
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

                // 计算当前 Shape 的包围盒（此时顶点如果是居中模式，已经被平移过了）
                lo = lo.minimum(vertices[idx0.vertex_index]);
                lo = lo.minimum(vertices[idx1.vertex_index]);
                lo = lo.minimum(vertices[idx2.vertex_index]);

                hi = hi.maximum(vertices[idx0.vertex_index]);
                hi = hi.maximum(vertices[idx1.vertex_index]);
                hi = hi.maximum(vertices[idx2.vertex_index]);

                if (idx0.vertex_index < shapeIds.size()) shapeIds[idx0.vertex_index] = sId;
                if (idx1.vertex_index < shapeIds.size()) shapeIds[idx1.vertex_index] = sId;
                if (idx2.vertex_index < shapeIds.size()) shapeIds[idx2.vertex_index] = sId;
            }
            tShapes[sId]->vertexIndex.assign(vertexIndex);
            tShapes[sId]->normalIndex.assign(normalIndex);
            tShapes[sId]->texCoordIndex.assign(texCoordIndex);

            tShapes[sId]->boundingBox = TAlignedBox3D<Real>(lo, hi);
            tShapes[sId]->boundingTransform = Transform3f(globalCenter, Quat1f().toMatrix3x3(), Vec3f(1.0f));

            sId++;
        }

        texMesh->geometry()->vertices().assign(vertices);
        texMesh->geometry()->normals().assign(normals);
        texMesh->geometry()->texCoords().assign(texCoords);
        texMesh->geometry()->shapeIds().assign(shapeIds);

        //A hack: for an obj file with one shape
        if (shapes.size() == 1)
        {
            texMesh->geometry()->shapeIds().resize(vertices.size());
            texMesh->geometry()->shapeIds().reset();
        }

        vertices.clear();
        normals.clear();
        texCoords.clear();

        return true;
    }

    bool loadObj(std::vector<Vec3f>& points, std::vector<TopologyModule::Triangle>& triangles, std::string filename, bool append)
    {
        if (!append)
        {
            points.clear();
            triangles.clear();
        }
        int offset = append ? points.size() : 0;

        tinyobj::attrib_t myattrib;
        std::vector <tinyobj::shape_t> myshape;
        std::vector <tinyobj::material_t> mymat;
        std::string mywarn;
        std::string myerr;

        char* fname = (char*)filename.c_str();
        std::cout << fname << std::endl;
        tinyobj::LoadObj(&myattrib, &myshape, &mymat, &mywarn, &myerr, fname, nullptr, true, true);
        std::cout << mywarn << std::endl;
        std::cout << myerr << std::endl;

        std::cout << "************************ Loading : shapelod    ************************ " << std::endl << std::endl;
        std::cout << "                        " << "    shape size =" << myshape.size() << std::endl << std::endl;
        std::cout << "************************ Loading : v    ************************ " << std::endl << std::endl;
        std::cout << "                        " << "    point sizelod = " << myattrib.vertices.size() / 3 << std::endl << std::endl;

        if (myshape.size() == 0) { return false; }

        for (int i = 0; i < myattrib.vertices.size() / 3; i++)
        {
            points.push_back(Vec3f(myattrib.vertices[3 * i], myattrib.vertices[3 * i + 1], myattrib.vertices[3 * i + 2]));
        }
        std::cout << "************************ Loading : f    ************************ " << std::endl << std::endl;
        for (int i = 0; i < myshape.size(); i++)
        {
            std::cout << "                        " << "    Triangle " << i << " size =" << myshape[i].mesh.indices.size() / 3 << std::endl << std::endl;

            for (int s = 0; s < myshape[i].mesh.indices.size() / 3; s++)
            {
                triangles.push_back(TopologyModule::Triangle(myshape[i].mesh.indices[3 * s].vertex_index + offset, myshape[i].mesh.indices[3 * s + 1].vertex_index + offset, myshape[i].mesh.indices[3 * s + 2].vertex_index + offset));
            }
        }
        std::cout << "************************ Loading completed    **********************" << std::endl << std::endl;
        return true;

    }

    bool manualParseSceneConfig(
		const std::string& xmlPath,
		std::vector<SceneObject>& sceneObjects,
		std::vector<Asset>& assets,
		std::vector<SceneJoint>* sceneJoints)
	{
		sceneObjects.clear();
		assets.clear();
		if (sceneJoints != nullptr)
		{
			sceneJoints->clear();
		}

		tinyxml2::XMLDocument doc;
		tinyxml2::XMLError loadResult = doc.LoadFile(xmlPath.c_str());
		if (loadResult != tinyxml2::XML_SUCCESS)
		{
			std::cerr << "Error: Could not load XML file: " << xmlPath << std::endl;
			return false;
		}

		tinyxml2::XMLElement* sceneElement = doc.FirstChildElement("Scene");
		if (sceneElement == nullptr)
		{
			std::cerr << "Error: Could not find <Scene> element in the XML file." << std::endl;
			return false;
		}

		std::map<std::string, int> assetIdToIndexMap;

		tinyxml2::XMLElement* assetsElement = sceneElement->FirstChildElement("Assets");
		if (assetsElement != nullptr)
		{
			int currentIndex = 0;
			for (tinyxml2::XMLElement* assetElement = assetsElement->FirstChildElement("Asset"); assetElement != nullptr; assetElement = assetElement->NextSiblingElement("Asset"))
			{
				Asset currentAsset;

				const char* assetId = assetElement->Attribute("id");
				if (assetId != nullptr)
				{
					currentAsset.name = assetId;
					assetIdToIndexMap[currentAsset.name] = currentIndex;
				}

				tinyxml2::XMLElement* modelElement = assetElement->FirstChildElement("Model");
				if (modelElement != nullptr && modelElement->GetText() != nullptr)
				{
					currentAsset.modelPath = modelElement->GetText();
				}

				tinyxml2::XMLElement* matElement = assetElement->FirstChildElement("Mat");
				if (matElement != nullptr && matElement->GetText() != nullptr)
				{
					currentAsset.matPath = matElement->GetText();
				}

				currentAsset.collisionProxy = parseSceneCollisionProxyElement(assetElement->FirstChildElement("CollisionProxy"));
				if (currentAsset.collisionProxy == SceneCollisionProxyType::Auto)
				{
					currentAsset.collisionProxy = parseSceneCollisionProxyElement(assetElement->FirstChildElement("CollisionShape"));
				}

				assets.push_back(currentAsset);
				currentIndex++;
			}
		}

		for (tinyxml2::XMLElement* objectElement = sceneElement->FirstChildElement("Object"); objectElement != nullptr; objectElement = objectElement->NextSiblingElement("Object"))
		{
			SceneObject currentObject;

			const char* objectName = objectElement->Attribute("name");
			currentObject.name = objectName != nullptr ? objectName : "";

			const char* assetIdStr = objectElement->Attribute("asset_id");
			if (assetIdStr != nullptr && assetIdToIndexMap.count(assetIdStr) > 0)
			{
				currentObject.asset_id = assetIdToIndexMap[assetIdStr];
			}
			else
			{
				std::cerr << "Warning: Object '" << currentObject.name << "' has an invalid or missing asset_id." << std::endl;
				currentObject.asset_id = -1;
			}

			tinyxml2::XMLElement* physics = objectElement->FirstChildElement("Physics");
			if (physics != nullptr)
			{
				tinyxml2::XMLElement* density = physics->FirstChildElement("Density");
				if (density != nullptr)
				{
					density->QueryFloatText(&currentObject.density);
				}

				tinyxml2::XMLElement* motionType = physics->FirstChildElement("MotionType");
				if (motionType != nullptr)
				{
					const char* motionTypeText = motionType->Attribute("value");
					if (motionTypeText == nullptr)
					{
						motionTypeText = motionType->GetText();
					}
					currentObject.motionType = parseSceneMotionType(motionTypeText);
				}

				currentObject.collisionProxy = parseSceneCollisionProxyElement(physics->FirstChildElement("CollisionProxy"));
				if (currentObject.collisionProxy == SceneCollisionProxyType::Auto)
				{
					currentObject.collisionProxy = parseSceneCollisionProxyElement(physics->FirstChildElement("CollisionShape"));
				}

				tinyxml2::XMLElement* initialVel = physics->FirstChildElement("InitialVelocity");
				if (initialVel != nullptr)
				{
					queryVec3Attributes(initialVel->FirstChildElement("Linear"), currentObject.linearVelocity);
					queryVec3Attributes(initialVel->FirstChildElement("Angular"), currentObject.angularVelocity);
				}
			}

			tinyxml2::XMLElement* transform = objectElement->FirstChildElement("Transform");
			if (transform != nullptr)
			{
				queryVec3Attributes(transform->FirstChildElement("Position"), currentObject.position);

				tinyxml2::XMLElement* orient = transform->FirstChildElement("Orientation");
				if (orient != nullptr)
				{
					orient->QueryFloatAttribute("pitch", &currentObject.orientation.x);
					orient->QueryFloatAttribute("yaw", &currentObject.orientation.y);
					orient->QueryFloatAttribute("roll", &currentObject.orientation.z);
				}

				queryVec3Attributes(transform->FirstChildElement("Scale"), currentObject.scale);
			}

			sceneObjects.push_back(currentObject);
		}

		if (sceneJoints == nullptr)
		{
			return true;
		}

		tinyxml2::XMLElement* jointsElement = sceneElement->FirstChildElement("Joints");
		if (jointsElement == nullptr)
		{
			return true;
		}

		for (tinyxml2::XMLElement* jointElement = jointsElement->FirstChildElement("Joint"); jointElement != nullptr; jointElement = jointElement->NextSiblingElement("Joint"))
		{
			SceneJoint currentJoint;

			const char* jointName = jointElement->Attribute("name");
			currentJoint.name = jointName != nullptr ? jointName : "";

			const char* jointType = jointElement->Attribute("type");
			if (jointType == nullptr)
			{
				tinyxml2::XMLElement* typeElement = jointElement->FirstChildElement("Type");
				if (typeElement != nullptr)
				{
					jointType = typeElement->GetText();
				}
			}
			currentJoint.type = parseSceneJointType(jointType);
			if (currentJoint.type == SceneJointType::Unknown)
			{
				std::cerr << "Warning: Joint '" << currentJoint.name << "' has an unsupported type." << std::endl;
				continue;
			}

			currentJoint.body1Name = readObjectRef(jointElement->FirstChildElement("Body1"));
			currentJoint.body2Name = readObjectRef(jointElement->FirstChildElement("Body2"));
			currentJoint.body2IsWorld = isWorldRef(currentJoint.body2Name);
			if (currentJoint.body2IsWorld)
			{
				currentJoint.body2Name.clear();
			}

			tinyxml2::XMLElement* anchorElement = jointElement->FirstChildElement("Anchor");
			if (anchorElement != nullptr)
			{
				currentJoint.hasAnchor = true;
				queryVec3Attributes(anchorElement, currentJoint.anchorPoint);
			}

			tinyxml2::XMLElement* axisElement = jointElement->FirstChildElement("Axis");
			if (axisElement != nullptr)
			{
				currentJoint.hasAxis = true;
				queryVec3Attributes(axisElement, currentJoint.axis);
			}

			tinyxml2::XMLElement* rangeElement = jointElement->FirstChildElement("Range");
			if (rangeElement != nullptr)
			{
				currentJoint.useRange = true;
				rangeElement->QueryFloatAttribute("min", &currentJoint.minValue);
				rangeElement->QueryFloatAttribute("max", &currentJoint.maxValue);
			}

			tinyxml2::XMLElement* motorElement = jointElement->FirstChildElement("Motor");
			if (motorElement != nullptr)
			{
				currentJoint.useMotor = true;
				if (motorElement->QueryFloatAttribute("value", &currentJoint.motorValue) != tinyxml2::XML_SUCCESS)
				{
					motorElement->QueryFloatAttribute("speed", &currentJoint.motorValue);
				}
			}

			if (currentJoint.type == SceneJointType::Point)
			{
				currentJoint.body2IsWorld = true;
				currentJoint.body2Name.clear();
			}

			if (currentJoint.body1Name.empty())
			{
				std::cerr << "Warning: Joint '" << currentJoint.name << "' is missing Body1." << std::endl;
				continue;
			}

			sceneJoints->push_back(currentJoint);
		}

		return true;
	}

    void computeMassProperties(const std::vector<Vec3f>& vertices, const std::vector<TopologyModule::Triangle>& faces, Real& out_volume, Vec3f& out_center, Mat3f& out_inertia)
    {
        Real total_volume = 0.0;
        Vec3f com_accumulator(0);
        Mat3f inertia_accumulator(0);

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

        if (total_volume < 0)
        {
            total_volume = -total_volume;
            com_accumulator = com_accumulator * Real(-1);
            inertia_accumulator = inertia_accumulator * Real(-1);
        }

        if (std::abs(total_volume) > 1e-9) {
            out_center = com_accumulator / total_volume;
        }
        else {
            Vec3f avg(0);
            for (const auto& v : vertices) avg += v;
            if (!vertices.empty()) avg /= vertices.size();
            out_center = avg;
        }

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
        std::vector<dyno::uint>  allShapeIds;

        auto& tShapes = texMesh->shapes();

        bool loadedSomething = false;

        for (auto& asset : assets)
        {
            const std::string& filename = getAssetPath() + asset.modelPath; // 假设 getAssetPath() 已定义
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

            std::string root = filePath.parent_path().string();

            bool result = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                filename.c_str(), root.c_str());

            if (!err.empty()) std::cerr << "Error: " << err << std::endl;
            if (!result) continue;

            loadedSomething = true;

            const uint32_t vertexOffset = static_cast<uint32_t>(allVertices.size());
            const uint32_t normalOffset = static_cast<uint32_t>(allNormals.size());
            const uint32_t texCoordOffset = static_cast<uint32_t>(allTexCoords.size());

            std::vector<dyno::Vec3f> currentVertices;
            std::vector<dyno::uint> currentShapeIds(attrib.vertices.size() / 3, 0);

            for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
                currentVertices.push_back({ attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2] });
            }
            for (size_t i = 0; i < attrib.normals.size(); i += 3) {
                allNormals.push_back({ attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2] });
            }
            for (size_t i = 0; i < attrib.texcoords.size(); i += 2) {
                allTexCoords.push_back({ attrib.texcoords[i], attrib.texcoords[i + 1] });
            }

            std::vector<std::shared_ptr<Material>> currentLocalMaterials;
            for (const auto& mtl : materials) {
                auto newMat = std::make_shared<Material>();

                newMat->baseColor = Color(mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2]);

                std::shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>();
                dyno::CArray2D<dyno::Vec4f> texture;

                if (!mtl.diffuse_texname.empty()) {
                    auto tex_path = (fs::path(root) / mtl.diffuse_texname).string();
                    if (loader->loadImage(tex_path.c_str(), texture)) newMat->texColor.assign(texture);
                }
                if (!mtl.bump_texname.empty()) {
                    auto tex_path = (fs::path(root) / mtl.bump_texname).string();
                    if (loader->loadImage(tex_path.c_str(), texture)) {
                        newMat->texBump.assign(texture);
                        newMat->bumpScale = mtl.bump_texopt.bump_multiplier;
                    }
                }
                currentLocalMaterials.push_back(newMat);
            }

            std::vector<TopologyModule::Triangle> assetLocalFaces;
            for (const tinyobj::shape_t& shape : shapes) {
                for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
                    assetLocalFaces.push_back({
                        shape.mesh.indices[i].vertex_index,
                        shape.mesh.indices[i + 1].vertex_index,
                        shape.mesh.indices[i + 2].vertex_index
                        });
                }
            }

			computeMassProperties(currentVertices, assetLocalFaces, asset.volume, asset.baryCenter, asset.inertialMatrix);

			Vec3f assetCenter = doTransform ? asset.baryCenter : Vec3f(0.0f);
			if (doTransform) {
				for (auto& v : currentVertices) {
					v -= assetCenter;
				}
			}

			if (!currentVertices.empty())
			{
				Vec3f localMin(REAL_MAX);
				Vec3f localMax(-REAL_MAX);
				for (const auto& v : currentVertices)
				{
					localMin = localMin.minimum(v);
					localMax = localMax.maximum(v);
				}
				asset.localBoundsMin = localMin;
				asset.localBoundsMax = localMax;
			}
			else
			{
				asset.localBoundsMin = Vec3f(0);
				asset.localBoundsMax = Vec3f(0);
			}

			for (const tinyobj::shape_t& shape : shapes) {
                const auto& mesh = shape.mesh;
                auto newShape = std::make_shared<Shape>();

                uint currentSId = static_cast<uint>(tShapes.size());

                std::vector<TopologyModule::Triangle> vertexIndex;
                std::vector<TopologyModule::Triangle> normalIndex;
                std::vector<TopologyModule::Triangle> texCoordIndex;

                if (!mesh.material_ids.empty() && mesh.material_ids[0] >= 0) {
                    if (mesh.material_ids[0] < currentLocalMaterials.size())
                        newShape->material = currentLocalMaterials[mesh.material_ids[0]];
                }

                Vec3f lo(REAL_MAX);
                Vec3f hi(-REAL_MAX);

                for (const auto& index : mesh.indices) {
                    lo = lo.minimum(currentVertices[index.vertex_index]);
                    hi = hi.maximum(currentVertices[index.vertex_index]);
                    if (index.vertex_index < currentShapeIds.size()) {
                        currentShapeIds[index.vertex_index] = currentSId;
                    }
                }

                for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                    auto idx0 = mesh.indices[i];
                    auto idx1 = mesh.indices[i + 1];
                    auto idx2 = mesh.indices[i + 2];

                    vertexIndex.push_back({
                        static_cast<int>(vertexOffset + idx0.vertex_index),
                        static_cast<int>(vertexOffset + idx1.vertex_index),
                        static_cast<int>(vertexOffset + idx2.vertex_index) });

                    if (idx0.normal_index >= 0) {
                        normalIndex.push_back({
                            static_cast<int>(normalOffset + idx0.normal_index),
                            static_cast<int>(normalOffset + idx1.normal_index),
                            static_cast<int>(normalOffset + idx2.normal_index) });
                    }
                    if (idx0.texcoord_index >= 0) {
                        texCoordIndex.push_back({
                            static_cast<int>(texCoordOffset + idx0.texcoord_index),
                            static_cast<int>(texCoordOffset + idx1.texcoord_index),
                            static_cast<int>(texCoordOffset + idx2.texcoord_index) });
                    }
                }

                newShape->boundingBox = TAlignedBox3D<Real>(lo, hi);
                newShape->boundingTransform = Transform3f(assetCenter, Quat1f().toMatrix3x3(), Vec3f(1.0f));

                newShape->vertexIndex.assign(vertexIndex);
                newShape->normalIndex.assign(normalIndex);
                newShape->texCoordIndex.assign(texCoordIndex);

                tShapes.push_back(newShape);
            }

            allVertices.insert(allVertices.end(), currentVertices.begin(), currentVertices.end());
            allShapeIds.insert(allShapeIds.end(), currentShapeIds.begin(), currentShapeIds.end());
        }

        if (loadedSomething) {
            texMesh->geometry()->vertices().assign(allVertices);
            texMesh->geometry()->normals().assign(allNormals);
            texMesh->geometry()->texCoords().assign(allTexCoords);
            texMesh->geometry()->shapeIds().assign(allShapeIds);
        }

        return loadedSomething;
    }
}
