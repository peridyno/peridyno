#pragma once
#include "Module/TopologyModule.h"

#include "AnimationCurve.h"
#include <iterator>
#include <random>
#include "TextureMesh.h"
#include "Quat.h"

#define ERRORTIME -2321.51

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
		Mat4f localTransform = Mat4f::identityMatrix();
		Vec3f localTranslation = Vec3f(0);
		Vec3f localRotation = Vec3f(0);
		Vec3f localScale = Vec3f(1);
		Vec3f preRotation = Vec3f(0);
		Vec3f pivot = Vec3f(0);

		std::vector<std::shared_ptr<ModelObject>> child;
		std::vector<std::shared_ptr<ModelObject>> parent;// bone - parent - root

		int id = -1;

		std::vector<Real> m_Translation_Times[3];
		std::vector<Real> m_Translation_Values[3];

		std::vector<Real> m_Rotation_Times[3];
		std::vector<Real> m_Rotation_Values[3];

		std::vector<Real> m_Scale_Times[3];
		std::vector<Real> m_Scale_Values[3];

		Vec3f getFrameTranslation(Real time);
		Vec3f getFrameRotation(Real time);
		Vec3f getFrameScale(Real time);

		
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
		std::map<int,std::vector<int>> pointId_verticeId;
		std::vector<Vec3f> normals;
		std::vector<Vec2f> texcoords;
		std::vector<Vec3f> verticesColor;
		std::vector<std::vector<TopologyModule::Triangle>> facegroup_triangles;
		std::vector<CArrayList<uint>> facegroup_polygons;

		std::vector<std::shared_ptr<Material>> materials;

		std::vector<TAlignedBox3D<Real>> boundingBox;
		std::vector<Transform3f> boundingTransform;

		std::vector<Vec4f> boneIndices0;
		std::vector<Vec4f> boneWeights0;
		std::vector<Vec4f> boneIndices1;
		std::vector<Vec4f> boneWeights1;		
		std::vector<Vec4f> boneIndices2;
		std::vector<Vec4f> boneWeights2;

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

		int getBoneIndexByName(std::string name) 
		{
			int id = 0;
			for (auto it : mBones) {
				if (it->name == name)
					return id;

				id++;
			}
			return -1;
		}


		void updateInverseBindMatrix() 
		{
			for (auto it : mBones)
			{
				//build inverseBindMatrix
				std::cout << "********************" << it->name << "\n";
				int select = getBoneIndexByName(it->name);
				if (select == -1)continue;

				Mat4f inverseMatrix = it->localTransform.inverse();
				for (size_t i = 0; i < it->parent.size(); i++)
				{
					auto parent = it->parent[i];
					inverseMatrix *= parent->localTransform.inverse();
					std::cout << parent->name << "\n";
				};

				mBoneInverseBindMatrix[select] = inverseMatrix;
			}
		}

		void updateFrameWorldTransform(Real time) //Ê±¼ä²åÖµ
		{
			
			//update Animation to mBoneRotations/mBoneTranslations/mBoneScales
			for (size_t i = 0; i < mBones.size(); i++)
			{
				int select = getBoneIndexByName(mBones[i]->name);
				if (select == -1)continue;

				auto iterR = mBones[select];

				//Rotation
				mBoneRotations[select].x = getVectorDataByTime(mBones[select]->m_Rotation_Values[0], mBones[select]->m_Rotation_Times[0], time);
				mBoneRotations[select].y = getVectorDataByTime(mBones[select]->m_Rotation_Values[1], mBones[select]->m_Rotation_Times[1], time);
				mBoneRotations[select].z = getVectorDataByTime(mBones[select]->m_Rotation_Values[2], mBones[select]->m_Rotation_Times[2], time);
				//Translation
				mBoneTranslations[select].x = getVectorDataByTime(mBones[select]->m_Translation_Values[0], mBones[select]->m_Translation_Times[0], time);
				mBoneTranslations[select].y = getVectorDataByTime(mBones[select]->m_Translation_Values[1], mBones[select]->m_Translation_Times[1], time);
				mBoneTranslations[select].z = getVectorDataByTime(mBones[select]->m_Translation_Values[2], mBones[select]->m_Translation_Times[2], time);
				//Scale
				mBoneScales[select].x = getVectorDataByTime(mBones[select]->m_Scale_Values[0], mBones[select]->m_Scale_Times[0], time);
				mBoneScales[select].y = getVectorDataByTime(mBones[select]->m_Scale_Values[1], mBones[select]->m_Scale_Times[1], time);
				mBoneScales[select].z = getVectorDataByTime(mBones[select]->m_Scale_Values[2], mBones[select]->m_Scale_Times[2], time);
			}

			for (auto it : mBones)
			{
				int select = getBoneIndexByName(it->name);
				if (select == -1)continue;

				Mat4f worldMatrix = it->localTransform;
				for (size_t i = 0; i < it->parent.size(); i++){
					auto parent = it->parent[i];
					worldMatrix *= parent->localTransform;
				};
				
				mBoneWorldMatrix[select] = worldMatrix;
			}
			currentTime = time;
			
		};

		Real getVectorDataByTime(std::vector<Real> data,std::vector<Real> timeCode,Real time)
		{
			if (!bool(data.size()))
				return 0;

			int idx = findMaxSmallerIndex(timeCode, time);
			if (idx >= data.size() - 1) {				//   [size-1]<=[tId]  
				return data[data.size() - 1];
			}
			else if(idx >= 0) {
				if (data[idx] != data[idx + 1]) {
					float weight = (time - timeCode[idx]) / (timeCode[idx + 1] - timeCode[idx]);
					return lerp(data[idx], data[idx + 1], weight);
				}
				else
					return data[idx];
			}
			else {
				return data[0];
			}
		}

		int findMaxSmallerIndex(const std::vector<float>& arr, float v) {
			int left = 0;
			int right = arr.size() - 1;
			int maxIndex = -1;

			if (arr.size() >= 1)
			{
				if (arr[0] > v)
					return 0;

				if (arr[arr.size() - 1] < v)
					return arr.size() - 1;
			}

			while (left <= right) {
				int mid = left + (right - left) / 2;

				if (arr[mid] <= v) {
					maxIndex = mid;
					left = mid + 1;
				}
				else {
					right = mid - 1;
				}
			}

			return maxIndex;
		}

	public:

		std::vector<std::shared_ptr<ModelObject>> mModelObjects;
		std::vector<std::shared_ptr<MeshInfo>> mMeshs;
		std::vector<std::shared_ptr<Bone>> mBones;;
		std::vector<Vec3f> mBoneRotations;
		std::vector<Vec3f> mBoneTranslations;
		std::vector<Vec3f> mBoneScales;
		std::vector<Mat4f> mBoneWorldMatrix;
		std::vector<Mat4f> mBoneInverseBindMatrix;

		float mTimeStart = -1;
		float mTimeEnd = -1;


	private:
		Real lerp(Real v0, Real v1, float weight)
		{
			return v0 + (v1 - v0) * weight;
		}
		Real currentTime = ERRORTIME;
	};
	

}