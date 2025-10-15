#pragma once
#include "Topology/TextureMesh.h"

#include <Field/FilePath.h>

#include "tinyxml/tinyxml2.h"

namespace dyno
{
	struct SceneObject {
		std::string name;
		uint asset_id;
		//Physics
		Real density;
		Vec3f linearVelocity;
		Vec3f angularVelocity;
		//Transform
		Vec3f position;
		Vec3f orientation;
		Vec3f scale;
	};

	struct Asset {
		std::string name;
		std::string modelPath;
		std::string matPath;
		Vec3f baryCenter;
		Mat3f inertialMatrix;
		Real volume;
	};

	bool loadTextureMeshFromObj(std::shared_ptr<TextureMesh> texMesh, const FilePath& fullname,bool useToCenter = true);

	bool manualParseSceneConfig(const std::string& xmlPath, std::vector<SceneObject>& sceneObjects, std::vector<Asset>& assets);

	bool loadObjects(std::shared_ptr<TextureMesh> texMesh, std::vector<Asset>& assets, std::vector<SceneObject>& sceneObjects, bool doTransform = true);
	
	bool loadObj(std::vector<Vec3f>& points, std::vector<TopologyModule::Triangle>& triangles, std::string filename, bool append = false);
	
}
