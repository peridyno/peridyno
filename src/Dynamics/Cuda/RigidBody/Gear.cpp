#include "Gear.h"

#include "Mapping/DiscreteElementsToTriangleSet.h"
#include "GLSurfaceVisualModule.h"
#include <cstddef>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "helpers/tinyobj_helper.h"

namespace dyno
{
	namespace
	{
		Vec3f absVec3(const Vec3f& value)
		{
			return Vec3f(std::abs(value.x), std::abs(value.y), std::abs(value.z));
		}

		Vec3f uniformSceneScale(const Vec3f& scale, const std::string& objectName)
		{
			const float sx = std::abs(scale.x);
			const float sy = std::abs(scale.y);
			const float sz = std::abs(scale.z);
			const float maxScale = std::max(sx, std::max(sy, sz));
			const float minScale = std::min(sx, std::min(sy, sz));

			if (maxScale <= 1.0e-6f)
			{
				std::cerr << "Warning: Object '" << objectName << "' has a zero scale. Fallback to 1." << std::endl;
				return Vec3f(1.0f);
			}

			if (maxScale - minScale > 1.0e-5f)
			{
				std::cerr << "Warning: Object '" << objectName
					<< "' uses non-uniform Scale, but MAT currently applies uniform scaling only." << std::endl;
			}

			return Vec3f(maxScale);
		}

		BodyType toBodyType(SceneMotionType motionType)
		{
			switch (motionType)
			{
			case SceneMotionType::Static:
				return BodyType::Static;
			case SceneMotionType::Kinematic:
				return BodyType::Kinematic;
			case SceneMotionType::Dynamic:
			default:
				return BodyType::Dynamic;
			}
		}

		Quat1f toSceneQuaternion(const Vec3f& orientationInDegrees)
		{
			return Quat1f(float(M_PI) * orientationInDegrees.z / 180.0f, Vec3f(0, 0, 1))
				* Quat1f(float(M_PI) * orientationInDegrees.y / 180.0f, Vec3f(0, 1, 0))
				* Quat1f(float(M_PI) * orientationInDegrees.x / 180.0f, Vec3f(1, 0, 0));
		}

		SceneCollisionProxyType resolveCollisionProxy(const SceneObject& object, const Asset& asset)
		{
			if (object.collisionProxy != SceneCollisionProxyType::Auto)
			{
				return object.collisionProxy;
			}
			if (asset.collisionProxy != SceneCollisionProxyType::Auto)
			{
				return asset.collisionProxy;
			}
			return asset.matPath.empty() ? SceneCollisionProxyType::Box : SceneCollisionProxyType::Mat;
		}

		BoxInfo buildAssetAlignedBox(const Asset& asset, const Vec3f& objectScale)
		{
			const Vec3f signedScale = objectScale;
			const Vec3f absScale = absVec3(objectScale);
			const Vec3f localCenter = (asset.localBoundsMin + asset.localBoundsMax) * 0.5f;
			const Vec3f localHalfExtent = (asset.localBoundsMax - asset.localBoundsMin) * 0.5f;

			BoxInfo box;
			box.center = Vec3f(
				localCenter.x * signedScale.x,
				localCenter.y * signedScale.y,
				localCenter.z * signedScale.z);
			box.halfLength = Vec3f(
				localHalfExtent.x * absScale.x,
				localHalfExtent.y * absScale.y,
				localHalfExtent.z * absScale.z);
			box.rot = Quat1f(0.0f, 0.0f, 0.0f, 1.0f);
			return box;
		}

		std::string extractChainCollisionGroup(const std::string& objectName)
		{
			const std::size_t linkPos = objectName.rfind("_link");
			if (linkPos == std::string::npos)
			{
				return std::string();
			}

			const std::size_t chainPos = objectName.rfind("_chain", linkPos);
			if (chainPos == std::string::npos)
			{
				return std::string();
			}

			return objectName.substr(0, linkPos);
		}

		std::unordered_map<std::string, uint> buildCollisionGroupsFromJoints(
			const std::vector<SceneObject>& objects,
			const std::vector<SceneJoint>& joints)
		{
			std::unordered_map<std::string, std::vector<std::string>> adjacency;
			adjacency.reserve(objects.size());

			for (const auto& object : objects)
			{
				adjacency.emplace(object.name, std::vector<std::string>());
			}

			for (const auto& joint : joints)
			{
				if (joint.body1Name.empty()
					|| joint.body2IsWorld
					|| joint.type == SceneJointType::Point
					|| joint.body2Name.empty())
				{
					continue;
				}

				auto body1It = adjacency.find(joint.body1Name);
				auto body2It = adjacency.find(joint.body2Name);
				if (body1It == adjacency.end() || body2It == adjacency.end())
				{
					continue;
				}

				body1It->second.push_back(joint.body2Name);
				body2It->second.push_back(joint.body1Name);
			}

			std::unordered_map<std::string, uint> groupsByObjectName;
			groupsByObjectName.reserve(objects.size());
			std::unordered_map<std::string, uint> fallbackGroupsByName;

			std::unordered_set<std::string> visited;
			visited.reserve(objects.size());

			uint nextCollisionGroup = 1;
			std::vector<std::string> stack;
			stack.reserve(objects.size());

			for (const auto& object : objects)
			{
				if (visited.find(object.name) != visited.end())
				{
					continue;
				}

				auto adjacencyIt = adjacency.find(object.name);
				if (adjacencyIt == adjacency.end())
				{
					continue;
				}

				stack.clear();
				stack.push_back(object.name);
				visited.insert(object.name);

				std::vector<std::string> component;
				component.reserve(8);

				while (!stack.empty())
				{
					const std::string current = stack.back();
					stack.pop_back();
					component.push_back(current);

					for (const auto& neighbor : adjacency[current])
					{
						if (visited.insert(neighbor).second)
						{
							stack.push_back(neighbor);
						}
					}
				}

				if (component.size() <= 1)
				{
					const std::string fallbackGroup = extractChainCollisionGroup(object.name);
					if (!fallbackGroup.empty())
					{
						auto fallbackIt = fallbackGroupsByName.find(fallbackGroup);
						if (fallbackIt == fallbackGroupsByName.end())
						{
							fallbackIt = fallbackGroupsByName.emplace(fallbackGroup, nextCollisionGroup++).first;
						}
						groupsByObjectName[object.name] = fallbackIt->second;
					}
					continue;
				}

				const uint groupId = nextCollisionGroup++;
				for (const auto& name : component)
				{
					groupsByObjectName[name] = groupId;
				}
			}

			return groupsByObjectName;
		}
	}

	//Gear
	IMPLEMENT_TCLASS(Gear, TDataType)

		template<typename TDataType>
	Gear<TDataType>::Gear() :
		ArticulatedBody<TDataType>()
	{

		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		this->stateTopology()->connect(mapper->inDiscreteElements());
		this->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->varBaseColor()->setValue(Color(1, 1, 0));
		sRender->varAlpha()->setValue(0.5f);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);

	}

	template<typename TDataType>
	Gear<TDataType>::~Gear()
	{

	}

	template<typename TDataType>
	void Gear<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		std::string filename = getAssetPath() + "gear/gear_up.obj";
		if (this->varFilePath()->getValue() != filename)
		{
			this->varFilePath()->setValue(FilePath(filename));
		}

		//first gear
		RigidBodyInfo info;
		info.position = Vec3f(0, 1, 0);
		info.angularVelocity = Vec3f(1, 0, 0);
		info.motionType = BodyType::Kinematic;
		info.bodyId = 0;
		auto actor = this->createRigidBody(info);

		CapsuleInfo capsule;
		capsule.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
		capsule.radius = 0.05f;
		capsule.halfLength = 0.26f;

		float r = 0.798f;
		for (uint sec = 0; sec < 24; sec++)
		{
			float theta = sec * M_PI / 12 + 0.115;
			float y = r * sin(theta);
			float z = r * cos(theta);

			capsule.center = Vec3f(-0.042f, y, z);
			this->bindCapsule(actor, capsule);
		}

		this->bindShape(actor, Pair<uint, uint>(0, 0));

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
	}

	DEFINE_CLASS(Gear);


	IMPLEMENT_TCLASS(MatBody, TDataType)

	template<typename TDataType>
	MatBody<TDataType>::MatBody() :
		ArticulatedBody<TDataType>()
	{
		// Keep collision proxies out of the default render path. Drawing them on
		// top of the OBJ mesh causes depth fighting and flickering highlights.
	}

	template<typename TDataType>
	MatBody<TDataType>::~MatBody()
	{

	}

	template<typename TDataType>
	void MatBody<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();
		this->Vertices.clear();
		this->Edges.clear();
		this->Faces.clear();

		std::string filename = getAssetPath() + this->mXmlPath;
		if (this->varFilePath()->getValue() != filename)
		{
			this->varFilePath()->setValue(FilePath(filename));
		}

		std::unordered_map<std::string, std::shared_ptr<PdActor>> actorsByName;
		const std::unordered_map<std::string, uint> collisionGroupsByObjectName =
			buildCollisionGroupsFromJoints(this->mObjects, this->mJoints);
		std::vector<bool> matLoaded(this->mAssets.size(), false);

		for (int i = 0; i < this->mObjects.size(); i++)
		{
			auto& object = this->mObjects[i];
			if (object.asset_id < 0 || object.asset_id >= static_cast<int>(mAssets.size()))
			{
				std::cerr << "Warning: Skip object '" << object.name << "' due to invalid asset index." << std::endl;
				continue;
			}

			auto& currentAsset = mAssets[object.asset_id];
			SceneCollisionProxyType collisionProxy = resolveCollisionProxy(object, currentAsset);
			if (collisionProxy == SceneCollisionProxyType::Mat && currentAsset.matPath.empty())
			{
				std::cerr << "Warning: Object '" << object.name
					<< "' requests MAT collision, but asset '" << currentAsset.name
					<< "' has no MAT file. Fallback to box proxy." << std::endl;
				collisionProxy = SceneCollisionProxyType::Box;
			}

			RigidBodyInfo info;
			info.friction = this->varFrictionCoefficient()->getValue();
			info.position = object.position;
			info.angle = toSceneQuaternion(object.orientation);
			info.linearVelocity = object.linearVelocity;
			info.angularVelocity = object.angularVelocity;
			info.motionType = toBodyType(object.motionType);
			info.bodyId = i;

			const auto collisionGroupIt = collisionGroupsByObjectName.find(object.name);
			if (collisionGroupIt != collisionGroupsByObjectName.end())
			{
				info.collisionGroup = collisionGroupIt->second;
			}

			std::shared_ptr<PdActor> actor = nullptr;
			Vec3f renderScale = object.scale;
			if (collisionProxy == SceneCollisionProxyType::Box)
			{
				BoxInfo box = buildAssetAlignedBox(currentAsset, object.scale);
				actor = this->createRigidBody(info);
				if (actor != nullptr)
				{
					this->bindBox(actor, box, object.density);
				}
			}
			else
			{
				const Vec3f objectScale = uniformSceneScale(object.scale, object.name);
				const float scaleFactor = objectScale.x;
				bool matReady = matLoaded[object.asset_id]
					&& object.asset_id < this->Vertices.size()
					&& (!this->Edges[object.asset_id].empty() || !this->Faces[object.asset_id].empty());

				if (!matLoaded[object.asset_id])
				{
					matReady = loadMa(currentAsset.matPath, object.asset_id);
					matLoaded[object.asset_id] = true;
				}

				if (!matReady
					|| object.asset_id >= this->Vertices.size()
					|| (this->Edges[object.asset_id].empty() && this->Faces[object.asset_id].empty()))
				{
					std::cerr << "Warning: Object '" << object.name
						<< "' requested MAT collision, but MAT proxy could not be loaded from '"
						<< currentAsset.matPath << "'. Fallback to box proxy." << std::endl;

					BoxInfo box = buildAssetAlignedBox(currentAsset, object.scale);
					actor = this->createRigidBody(info);
					if (actor != nullptr)
					{
						this->bindBox(actor, box, object.density);
					}
				}
				else
				{
					const float scale2 = scaleFactor * scaleFactor;
					const float scale3 = scale2 * scaleFactor;
					const float scale5 = scale3 * scale2;
					MedialConeInfo medalcone;
					MedialSlabInfo medalslab;
					info.mass = currentAsset.volume * scale3 * object.density;
					info.inertia = currentAsset.inertialMatrix * (scale5 * object.density);
					actor = this->createRigidBody(info, false);
					renderScale = objectScale;

					if (actor != nullptr)
					{
						auto& vertices = this->Vertices[object.asset_id];
						auto& edges = this->Edges[object.asset_id];
						auto& faces = this->Faces[object.asset_id];
						for (size_t j = 0; j < edges.size(); j++)
						{
							Vec2i edge = edges[j];
							if (edge[0] >= vertices.size() || edge[1] >= vertices.size())
							{
								std::cerr << "ERROR load edge" << std::endl;
								continue;
							}
							medalcone.v[0] = (Vec3f(vertices[edge[0]][0], vertices[edge[0]][1], vertices[edge[0]][2]) - currentAsset.baryCenter) * scaleFactor;
							medalcone.v[1] = (Vec3f(vertices[edge[1]][0], vertices[edge[1]][1], vertices[edge[1]][2]) - currentAsset.baryCenter) * scaleFactor;
							medalcone.radius[0] = vertices[edge[0]][3] * scaleFactor;
							medalcone.radius[1] = vertices[edge[1]][3] * scaleFactor;
							this->bindMedialCone(actor, medalcone);
						}

						for (size_t j = 0; j < faces.size(); j++)
						{
							Vec3i face = faces[j];
							if (face[0] >= vertices.size() || face[1] >= vertices.size() || face[2] >= vertices.size())
							{
								std::cerr << "ERROR load face" << std::endl;
								continue;
							}
							medalslab.v[0] = (Vec3f(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2]) - currentAsset.baryCenter) * scaleFactor;
							medalslab.v[1] = (Vec3f(vertices[face[1]][0], vertices[face[1]][1], vertices[face[1]][2]) - currentAsset.baryCenter) * scaleFactor;
							medalslab.v[2] = (Vec3f(vertices[face[2]][0], vertices[face[2]][1], vertices[face[2]][2]) - currentAsset.baryCenter) * scaleFactor;
							medalslab.radius[0] = vertices[face[0]][3] * scaleFactor;
							medalslab.radius[1] = vertices[face[1]][3] * scaleFactor;
							medalslab.radius[2] = vertices[face[2]][3] * scaleFactor;
							this->bindMedialSlab(actor, medalslab);
						}
					}
				}
			}

			if (actor == nullptr)
			{
				continue;
			}
			actorsByName[object.name] = actor;
			this->bindShape(actor, Pair<uint, uint>(object.asset_id, i), renderScale);
		}

		for (const auto& jointInfo : this->mJoints)
		{
			auto actor1It = actorsByName.find(jointInfo.body1Name);
			if (actor1It == actorsByName.end())
			{
				std::cerr << "Warning: Joint '" << jointInfo.name << "' cannot resolve Body1 '" << jointInfo.body1Name << "'." << std::endl;
				continue;
			}

			auto actor1 = actor1It->second;
			std::shared_ptr<PdActor> actor2 = nullptr;
			if (!jointInfo.body2IsWorld && jointInfo.type != SceneJointType::Point)
			{
				auto actor2It = actorsByName.find(jointInfo.body2Name);
				if (actor2It == actorsByName.end())
				{
					std::cerr << "Warning: Joint '" << jointInfo.name << "' cannot resolve Body2 '" << jointInfo.body2Name << "'." << std::endl;
					continue;
				}
				actor2 = actor2It->second;
			}

			const Vec3f anchorPoint = jointInfo.hasAnchor ? jointInfo.anchorPoint : actor1->center;

			switch (jointInfo.type)
			{
			case SceneJointType::Fixed:
				if (jointInfo.body2IsWorld || actor2 == nullptr)
				{
					auto& joint = this->createUnilateralFixedJoint(actor1);
					joint.setAnchorPoint(anchorPoint);
				}
				else
				{
					auto& joint = this->createFixedJoint(actor1, actor2);
					joint.setAnchorPoint(anchorPoint);
				}
				break;
			case SceneJointType::Point:
			{
				auto& joint = this->createPointJoint(actor1);
				joint.setAnchorPoint(anchorPoint);
				break;
			}
			case SceneJointType::BallAndSocket:
				if (jointInfo.body2IsWorld || actor2 == nullptr)
				{
					std::cerr << "Warning: BallAndSocket joint '" << jointInfo.name << "' requires Body2." << std::endl;
					break;
				}
				else
				{
					auto& joint = this->createBallAndSocketJoint(actor1, actor2);
					joint.setAnchorPoint(anchorPoint);
				}
				break;
			case SceneJointType::Hinge:
				if (jointInfo.body2IsWorld || actor2 == nullptr)
				{
					std::cerr << "Warning: Hinge joint '" << jointInfo.name << "' requires Body2." << std::endl;
					break;
				}
				else
				{
					auto& joint = this->createHingeJoint(actor1, actor2);
					joint.setAnchorPoint(anchorPoint);
					joint.setAxis(jointInfo.axis);
					if (jointInfo.useMotor)
						joint.setMoter(jointInfo.motorValue);
					if (jointInfo.useRange)
						joint.setRange(jointInfo.minValue, jointInfo.maxValue);
				}
				break;
			case SceneJointType::Slider:
				if (jointInfo.body2IsWorld || actor2 == nullptr)
				{
					std::cerr << "Warning: Slider joint '" << jointInfo.name << "' requires Body2." << std::endl;
					break;
				}
				else
				{
					auto& joint = this->createSliderJoint(actor1, actor2);
					joint.setAnchorPoint(anchorPoint);
					joint.setAxis(jointInfo.axis);
					if (jointInfo.useMotor)
						joint.setMoter(jointInfo.motorValue);
					if (jointInfo.useRange)
						joint.setRange(jointInfo.minValue, jointInfo.maxValue);
				}
				break;
			case SceneJointType::Unknown:
			default:
				std::cerr << "Warning: Ignore unsupported joint '" << jointInfo.name << "'." << std::endl;
				break;
			}
		}

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
	}

	template<typename TDataType>
	bool MatBody<TDataType>::loadMa(std::string file_path, int objectId)
	{
		std::vector<Vec4f> vertices;
		std::vector<Vec2i> edges;
		std::vector<Vec3i>	faces;
		std::string filename = getAssetPath() + file_path;
		std::ifstream inputFile(filename);
		std::string line;
		int num_vertices = 0, num_edges = 0, num_faces = 0;

		if (!inputFile.is_open())
		{
			std::cerr << "ERROR MA FILE: cannot open " << filename << std::endl;
			return false;
		}

		if (std::getline(inputFile, line))
		{
			std::istringstream iss(line);
			if (!(iss >> num_vertices >> num_edges >> num_faces)) {
				std::cerr << "ERROR MA FILE: invalid header in " << filename << std::endl;
				inputFile.close();
				return false;
			}
			std::cout << "num Of vertices : " << num_vertices
				<< ", num Of edges : " << num_edges
				<< ", num Of faces : " << num_faces << std::endl;
		}
		else
		{
			std::cerr << "ERROR MA FILE: empty file " << filename << std::endl;
			inputFile.close();
			return false;
		}

		int current_vertex_count = 0;
		int current_edge_count = 0;
		int current_face_count = 0;

		while (std::getline(inputFile, line))
		{
			if (line.empty() || line[0] == '#')
			{
				continue;
			}

			std::istringstream iss(line);
			char type;
			iss >> type;

			if (type == 'v' && current_vertex_count < num_vertices)
			{
				Vec4f vertex;
				if (iss >> vertex[0] >> vertex[1] >> vertex[2] >> vertex[3])
				{
					vertex = vertex;
					vertices.push_back(vertex);
					current_vertex_count++;
				}
				else {
					std::cerr << "ERROR load vertex" << std::endl;
				}
			}
			else if (type == 'e' && current_edge_count < num_edges)
			{
				Vec2i edge;
				if (iss >> edge[0] >> edge[1])
				{
					if (edge[0] >= 0 && edge[0] < num_vertices && edge[1] >= 0 && edge[1] < num_vertices)
					{
						edges.push_back(edge);
						current_edge_count++;
					}
					else {
						std::cerr << "ERROR load edge" << std::endl;
					}
				}
				else {
					std::cerr << "ERROR load edge" << std::endl;
				}
			}
			else if (type == 'f' && current_face_count < num_faces)
			{
				Vec3i face;
				if (iss >> face[0] >> face[1] >> face[2])
				{
					if (face[0] >= 0 && face[0] < num_vertices && face[1] >= 0 && face[1] < num_vertices && face[2] >= 0 && face[2] < num_vertices)
					{
						faces.push_back(face);
						current_face_count++;
					}
					else {
						std::cerr << "ERROR load face" << std::endl;
					}
				}
				else {
					std::cerr << "ERROR load face" << std::endl;
				}
			}
		}
		inputFile.close();

		if (objectId >= this->Vertices.size())
		{
			this->Vertices.resize(objectId + 1);
			this->Edges.resize(objectId + 1);
			this->Faces.resize(objectId + 1);
		}

		this->Vertices[objectId] = std::move(vertices);
		this->Edges[objectId] = std::move(edges);
		this->Faces[objectId] = std::move(faces);

		return num_vertices == current_vertex_count
			&& num_edges == current_edge_count
			&& num_faces == current_face_count;
	}

	DEFINE_CLASS(MatBody);
}
