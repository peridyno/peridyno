#pragma once
#include "Module/TopologyModule.h"

#include "AnimationCurve.h"
#include <iterator>
#include <random>
#include "TextureMesh.h"
#include "Quat.h"
#include "Topology/JointInfo.h"
#include "Topology/SkinInfo.h"

#define ERRORTIME -2321.51

namespace dyno
{
	class ModelObject;
	class Bone;

	class ModelObject : public Object
	{
	public:

		ModelObject() {};
		~ModelObject();
		bool operator==(const ModelObject& model);

	public:

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
	};


	class Bone : public ModelObject
	{
	public:

		Bone() {};

	public:

		Mat4f inverseBindMatrix = Mat4f::identityMatrix();
	};


	class MeshInfo : public ModelObject
	{
	public:
		MeshInfo();

		~MeshInfo();

		void resizeSkin(int size);

		unsigned int size() { return vertices.size(); }

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
		HierarchicalScene();
		
		~HierarchicalScene();

		void clear();

		int findMeshIndexByName(std::string name);
		int findObjectIndexByName(std::string name);
		void pushBackBone(std::shared_ptr<Bone> bone);
		void pushBackMesh(std::shared_ptr<MeshInfo> mesh);
		std::shared_ptr<ModelObject> getObjectByName(std::string name);
		int getObjIndexByName(std::string name);
		int getBoneIndexByName(std::string name);
		void updateWorldMatrix();
		void updateInverseBindMatrix();
		void updateWorldTransformByKeyFrame(Real time);
		Real getVectorDataByTime(std::vector<Real> data, std::vector<Real> timeCode, Real time);
		int findMaxSmallerIndex(const std::vector<float>& arr, float v);
		std::vector<std::shared_ptr<Bone>>& getBones() { return mBones; }
		void skinAnimation(
			DArray<Vec3f>& intialPosition,
			DArray<Vec3f>& worldPosition,
			DArray<Mat4f>& joint_inverseBindMatrix,
			DArray<Mat4f>& WorldMatrix,
			DArray<Vec4f>& bind_joints_0,
			DArray<Vec4f>& bind_joints_1,
			DArray<Vec4f>& bind_joints_2,
			DArray<Vec4f>& weights_0,
			DArray<Vec4f>& weights_1,
			DArray<Vec4f>& weights_2,
			Mat4f transform,
			bool isNormal,
			Vec2u range
		);

		void UpdateJointData();

		void coutBoneHierarchial();

		void updateSkinData(std::shared_ptr<TextureMesh> texMesh);

		Mat4f createLocalTransform(std::shared_ptr<ModelObject> object);

		void coutMatrix(int id, Mat4f c)
		{
			printf("********** step: %d  ***********\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n ***********************\n\n",
				id,
				c(0, 0), c(0, 1), c(0, 2), c(0, 3),
				c(1, 0), c(1, 1), c(1, 2), c(1, 3),
				c(2, 0), c(2, 1), c(2, 2), c(2, 3),
				c(3, 0), c(3, 1), c(3, 2), c(3, 3)
			);
		}

		void showJointInfo()
		{
			std::string str;

			for (auto it : mBones)
			{
				if (!it->parent.size()) 
				{
					str.append(it->name);
					buildTree(str,it->child,1);
				}
			}
			std::cout << str << "\n";
		}

	private :

		void buildTree(std::string& str, const std::vector<std::shared_ptr<ModelObject>>& child, uint level)
		{
			str.append("\n");

			for (auto chi : child)
			{
				for (size_t i = 0; i < level; i++)
				{
					str.append(" ");
				}
				str.append("-");
				str.append(chi->name);
				buildTree(str, chi->child,level + 1);
			}
		}

	public:

		std::vector<std::shared_ptr<ModelObject>> mModelObjects;
		std::vector<std::shared_ptr<MeshInfo>> mMeshs;
		std::vector<std::shared_ptr<Bone>> mBones;
		std::vector<Vec3f> mBoneRotations;
		std::vector<Vec3f> mBoneTranslations;
		std::vector<Vec3f> mBoneScales;
		std::vector<Mat4f> mBoneWorldMatrix;
		std::vector<Mat4f> mBoneInverseBindMatrix;
		std::vector<Mat4f> mBoneLocalMatrix;

		std::shared_ptr<JointInfo> mJointData;
		std::shared_ptr<SkinInfo> mSkinData;
		std::shared_ptr<JointAnimationInfo> mJointAnimationData;

		float mTimeStart = -1;
		float mTimeEnd = -1;


	private:
		Real lerp(Real v0, Real v1, float weight);
		Real currentTime = ERRORTIME;
	};
	
	
}