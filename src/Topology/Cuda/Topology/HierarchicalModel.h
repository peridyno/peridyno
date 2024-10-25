#pragma once
#include "Module/TopologyModule.h"

#include "AnimationCurve.h"
#include <iterator>
#include <random>

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

	class HierarchicalScene : public Object
	{
	public:
		HierarchicalScene() {}
		~HierarchicalScene() { clear(); }

		void clear() 
		{
			mModelObjects.clear();
		}

		std::vector<std::shared_ptr<ModelObject>> mModelObjects;

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

		float mTimeStart = -1;
		float mTimeEnd = -1;

	};
	

}