
#pragma once
#include "Topology/TextureMesh.h"

#include <Field/FilePath.h>

#include "tinyxml/tinyxml2.h"

namespace dyno
{
	enum class SceneMotionType
	{
		Static,
		Kinematic,
		Dynamic
	};

	enum class SceneCollisionProxyType
	{
		Auto,
		Box,
		Mat
	};

	enum class SceneJointType
	{
		BallAndSocket,
		Slider,
		Hinge,
		Fixed,
		Point,
		Unknown
	};

	struct SceneObject {
		std::string name;
		int asset_id = -1;
		//Physics
		Real density = Real(1);
		Vec3f linearVelocity = Vec3f(0);
		Vec3f angularVelocity = Vec3f(0);
		SceneMotionType motionType = SceneMotionType::Dynamic;
		SceneCollisionProxyType collisionProxy = SceneCollisionProxyType::Auto;
		//Transform
		Vec3f position = Vec3f(0);
		Vec3f orientation = Vec3f(0);
		Vec3f scale = Vec3f(1);
	};

	struct Asset {
		std::string name;
		std::string modelPath;
		std::string matPath;
		Vec3f baryCenter;
		Mat3f inertialMatrix;
		Real volume;
		SceneCollisionProxyType collisionProxy = SceneCollisionProxyType::Auto;
		Vec3f localBoundsMin = Vec3f(0);
		Vec3f localBoundsMax = Vec3f(0);
	};

	struct SceneJoint {
		std::string name;
		std::string body1Name;
		std::string body2Name;
		SceneJointType type = SceneJointType::Unknown;
		bool body2IsWorld = false;
		bool hasAnchor = false;
		bool hasAxis = false;
		bool useMotor = false;
		bool useRange = false;
		Vec3f anchorPoint = Vec3f(0);
		Vec3f axis = Vec3f(1, 0, 0);
		Real minValue = Real(0);
		Real maxValue = Real(0);
		Real motorValue = Real(0);
	};

	bool loadTextureMeshFromObj(std::shared_ptr<TextureMesh> texMesh, const FilePath& fullname, bool useToCenter = true);

	bool manualParseSceneConfig(
		const std::string& xmlPath,
		std::vector<SceneObject>& sceneObjects,
		std::vector<Asset>& assets,
		std::vector<SceneJoint>* sceneJoints = nullptr);

	bool loadObjects(std::shared_ptr<TextureMesh> texMesh, std::vector<Asset>& assets, std::vector<SceneObject>& sceneObjects, bool doTransform = true);

	bool loadObj(std::vector<Vec3f>& points, std::vector<TopologyModule::Triangle>& triangles, std::string filename, bool append = false);

}
