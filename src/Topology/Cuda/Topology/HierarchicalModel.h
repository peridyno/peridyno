#pragma once
#include "Module/TopologyModule.h"

#include "AnimationCurve.h"
#include <iterator>
#include <random>
#include "TextureMesh.h"
#include "Quat.h"

namespace dyno
{
	class ModelObject;
	class Bone;

	class ModelObject : public Object
	{
	public:
		ModelObject() {};
		~ModelObject() 
		{
			child.clear();
			parent.clear();


		};

		bool operator==(const ModelObject& model)
		{
			return name == model.name;
		}

		std::string name;
		Vec3f localTranslation = Vec3f(0);
		Vec3f localRotation = Vec3f(0);
		Vec3f localScale = Vec3f(1);
		Vec3f preRotation = Vec3f(0);
		Vec3f pivot = Vec3f(0);

		std::vector<std::shared_ptr<ModelObject>> child;
		std::vector<std::shared_ptr<ModelObject>> parent;

		int id = -1;

		std::vector<Real> m_Translation_Times[3];
		std::vector<Real> m_Translation_Values[3];

		std::vector<Real> m_Rotation_Times[3];
		std::vector<Real> m_Rotation_Values[3];

		std::vector<Real> m_Scale_Times[3];
		std::vector<Real> m_Scale_Values[3];
	};


	class Bone : public ModelObject
	{
	public:
		Bone() {};

	};

	class MeshInfo : public ModelObject
	{
	public:
		MeshInfo() {};

		std::vector<Vec3f> vertices;
		std::vector<int> verticeId_pointId;
		std::vector<Vec3f> normals;
		std::vector<Vec2f> texcoords;
		std::vector<Vec3f> verticesColor;
		std::vector<std::vector<TopologyModule::Triangle>> facegroup_triangles;
		std::vector<CArrayList<uint>> facegroup_polygons;

		std::vector<std::shared_ptr<Material>> materials;

		std::vector<TAlignedBox3D<Real>> boundingBox;
		std::vector<Transform3f> boundingTransform;
	};


	class HierarchicalScene : public Object
	{
	public:
		HierarchicalScene() 
		{
			mTimeStart = -1;
			mTimeEnd = -1;
		}
		~HierarchicalScene() { clear(); }

		void clear() 
		{
			mModelObjects.clear();
		}

		int findMeshIndexByName(std::string name)
		{
			int id = 0;
			for (auto it : mMeshs) {
				if (it->name == name)
					return id;

				id++;
			}
			return -1;
		}

		std::shared_ptr<ModelObject> getObjectByName(std::string name)
		{
			for (auto it : mModelObjects) {
				if (it->name == name) 
				{
					return it;
				}
			}

			return nullptr;
		}

		int getObjIndexByName(std::string name)
		{
			int id = 0;
			for (auto it : mModelObjects) {
				if (it->name == name)
					return id;

				id++;
			}
			return -1;
		}

	public:

		std::vector<std::shared_ptr<ModelObject>> mModelObjects;
		std::vector<std::shared_ptr<MeshInfo>> mMeshs;

		float mTimeStart = -1;
		float mTimeEnd = -1;

	};
	

}