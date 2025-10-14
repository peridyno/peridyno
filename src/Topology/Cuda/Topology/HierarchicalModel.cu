#include "HierarchicalModel.h"

namespace dyno
{
	template< typename Coord, typename Vec4f, typename Mat4f, typename Vec2u>
	__global__ void PointsAnimation(
		DArray<Coord> intialPosition,
		DArray<Coord> worldPosition,
		DArray<Mat4f> joint_inverseBindMatrix,
		DArray<Mat4f> WorldMatrix,

		DArray<Vec4f> bind_joints_0,
		DArray<Vec4f> bind_joints_1,
		DArray<Vec4f> bind_joints_2,
		DArray<Vec4f> weights_0,
		DArray<Vec4f> weights_1,
		DArray<Vec4f> weights_2,

		Mat4f transform,
		bool isNormal,

		Vec2u range
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= worldPosition.size()) return;

		if (pId<range[0] || pId>range[1])
			return;

		Coord initialP = intialPosition[pId];

		Vec4f result = Vec4f(0, 0, 0, float(!isNormal));

		int skinInfoVertexId = pId - range[0];

		Coord offest;

		bool j0 = bind_joints_0.size();
		bool j1 = bind_joints_1.size();
		bool j2 = bind_joints_2.size();

		if (j0)
		{
			for (unsigned int i = 0; i < 4; i++)
			{
				int jointId = int(bind_joints_0[skinInfoVertexId][i]);
				Real weight = weights_0[skinInfoVertexId][i];

				offest = intialPosition[pId];
				Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], float(!isNormal));//

				result += (transform * WorldMatrix[jointId] * v_bone_space) * weight;
			}
		}
		if (j1)
		{
			for (unsigned int i = 0; i < 4; i++)
			{
				int jointId = int(bind_joints_1[skinInfoVertexId][i]);
				Real weight = weights_1[skinInfoVertexId][i];

				offest = intialPosition[pId];
				Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], float(!isNormal));//

				result += (transform * WorldMatrix[jointId] * v_bone_space) * weight;
			}
		}
		if (j2)
		{
			for (unsigned int i = 0; i < 4; i++)
			{
				int jointId = int(bind_joints_2[skinInfoVertexId][i]);
				Real weight = weights_2[skinInfoVertexId][i];

				offest = intialPosition[pId];
				Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], float(!isNormal));//

				result += (transform * WorldMatrix[jointId] * v_bone_space) * weight;
			}
		}
		//result = transform * result;
		//printf("V : %f,%f,%f\nRange: %d,%d\n", result[0], result[1], result[2], range[0],range[1]);
		

		if (j0 || j1)
		{
			worldPosition[pId][0] = result[0];
			worldPosition[pId][1] = result[1];
			worldPosition[pId][2] = result[2];
		}

		if (isNormal)
			worldPosition[pId] = worldPosition[pId].normalize();
		//if(pId > 350000 && !isNormal)
			//printf("%d : %f,%f,%f - %f,%f,%f \n",pId, initialP[0], initialP[1], initialP[2], worldPosition[pId][0], worldPosition[pId][1], worldPosition[pId][2]);
	}

	void HierarchicalScene::skinAnimation(
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
	)
	{
		cuExecute(intialPosition.size(),
			PointsAnimation,
			intialPosition,
			worldPosition,
			joint_inverseBindMatrix,
			WorldMatrix,

			bind_joints_0,
			bind_joints_1,
			bind_joints_2,
			weights_0,
			weights_1,
			weights_2,
			transform,
			isNormal,
			range
		);

	}

	template< typename Coord, typename Vec4f, typename Mat4f, typename Vec2u>
	__global__ void verticesAnimation(
		DArray<Coord> intialVertices,
		DArray<Coord> Vertices,
		DArray<Mat4f> joint_inverseBindMatrix,
		DArray<Mat4f> WorldMatrix,

		DArrayList<int> point2Vertice,
		DArray<Vec4f> bind_joints_0,
		DArray<Vec4f> bind_joints_1,
		DArray<Vec4f> bind_joints_2,
		DArray<Vec4f> weights_0,
		DArray<Vec4f> weights_1,
		DArray<Vec4f> weights_2,

		Mat4f transform,
		bool isNormal,

		Vec2u range
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= point2Vertice.size()) return;

		if (pId < range[0] || pId >= range[1])
			return;

		List<int>& list = point2Vertice[pId];

		for (int k = 0; k < list.size(); k++)
		{
			int vId = list[k];
			Vec3f initialP = intialVertices[vId];

			Vec4f result = Vec4f(0, 0, 0, float(!isNormal));

			int skinPId = pId - range[0];

			Vec3f offest;

			bool j0 = bind_joints_0.size();
			bool j1 = bind_joints_1.size();
			bool j2 = bind_joints_2.size();

			if (j0)
			{
				for (unsigned int i = 0; i < 4; i++)
				{
					int jointId = int(bind_joints_0[skinPId][i]);
					Real weight = weights_0[skinPId][i];

					offest = intialVertices[vId];

					Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], float(!isNormal));//

					result += (transform * WorldMatrix[jointId] * v_bone_space) * weight;

				}
			}
			if (j1)
			{
				for (unsigned int i = 0; i < 4; i++)
				{
					int jointId = int(bind_joints_1[skinPId][i]);
					Real weight = weights_1[skinPId][i];

					offest = intialVertices[vId];
					Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], float(!isNormal));//

					result += (transform * WorldMatrix[jointId] * v_bone_space) * weight;


				}
			}
			if (j2)
			{
				for (unsigned int i = 0; i < 4; i++)
				{
					int jointId = int(bind_joints_2[skinPId][i]);
					Real weight = weights_2[skinPId][i];

					offest = intialVertices[vId];
					Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], float(!isNormal));//

					result += (transform * WorldMatrix[jointId] * v_bone_space) * weight;
				}
			}


			if (j0 || j1 || j2)
			{

				Vertices[vId][0] = result[0];
				Vertices[vId][1] = result[1];
				Vertices[vId][2] = result[2];
			}

			if (isNormal) 
			{
				Vertices[vId] = Vertices[vId].normalize();



			}
		}

	}

	void HierarchicalScene::skinVerticesAnimation(
		DArray<Vec3f>& intialVertices,
		DArray<Vec3f>& Vertices,
		DArray<Mat4f>& joint_inverseBindMatrix,
		DArray<Mat4f>& WorldMatrix,

		DArrayList<int>& point2Vertice,
		DArray<Vec4f>& bind_joints_0,
		DArray<Vec4f>& bind_joints_1,
		DArray<Vec4f>& bind_joints_2,
		DArray<Vec4f>& weights_0,
		DArray<Vec4f>& weights_1,
		DArray<Vec4f>& weights_2,

		Mat4f transform,
		bool isNormal,

		Vec2u range
	) 
	{
		cuExecute(point2Vertice.size(),
			verticesAnimation,
			intialVertices,
			Vertices,
			joint_inverseBindMatrix,
			WorldMatrix,
			point2Vertice,
			bind_joints_0,
			bind_joints_1,
			bind_joints_2,
			weights_0,
			weights_1,
			weights_2,
			transform,
			isNormal,
			range
		);
	}


	template< typename Vec3f, typename Vec4f, typename Mat4f, typename Vec2u>
	__global__ void GetVerticesNormalInBindPose(
		DArray<Vec3f> initialNormal,
		DArray<Mat4f> joint_inverseBindMatrix,
		DArray<Mat4f> WorldMatrix,

		DArrayList<int> point2Vertice,
		DArray<Vec4f> bind_joints_0,
		DArray<Vec4f> bind_joints_1,
		DArray<Vec4f> bind_joints_2,
		DArray<Vec4f> weights_0,
		DArray<Vec4f> weights_1,
		DArray<Vec4f> weights_2,

		Vec2u range
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= point2Vertice.size()) return;

		if (pId < range[0] || pId >= range[1])
			return;

		List<int>& list = point2Vertice[pId];

		for (int k = 0; k < list.size(); k++)
		{
			int vId = list[k];

			Vec3f N = initialNormal[vId];

			int skinPId = pId - range[0];

			bool j0 = bind_joints_0.size();
			bool j1 = bind_joints_1.size();
			bool j2 = bind_joints_2.size();

			Vec3f result = Vec3f(0);

			float add = 0;

			if (j0)
			{
				for (unsigned int i = 0; i < 4; i++)
				{
					int jointId = int(bind_joints_0[skinPId][i]);
					Real weight = weights_0[skinPId][i];

					Mat4f temp = WorldMatrix[jointId] * joint_inverseBindMatrix[jointId];
					Mat3f temp3 = Mat3f(temp(0, 0), temp(0, 1), temp(0, 2),
						temp(1, 0), temp(1, 1), temp(1, 2),
						temp(2, 0), temp(2, 1), temp(2, 2)
					);

					result += (temp3.inverse() * N) * weight;
					add += weight;
				}
			}
			if (j1)
			{
				for (unsigned int i = 0; i < 4; i++)
				{
					int jointId = int(bind_joints_1[skinPId][i]);
					Real weight = weights_1[skinPId][i];

					Mat4f temp = WorldMatrix[jointId] * joint_inverseBindMatrix[jointId];
					Mat3f temp3 = Mat3f(temp(0, 0), temp(0, 1), temp(0, 2),
						temp(1, 0), temp(1, 1), temp(1, 2),
						temp(2, 0), temp(2, 1), temp(2, 2)
					);

					result += (temp3.inverse() * N) * weight;
					add += weight;
				}
			}
			if (j2)
			{
				for (unsigned int i = 0; i < 4; i++)
				{
					int jointId = int(bind_joints_2[skinPId][i]);
					Real weight = weights_2[skinPId][i];

					Mat4f temp = WorldMatrix[jointId] * joint_inverseBindMatrix[jointId];
					Mat3f temp3 = Mat3f(temp(0, 0), temp(0, 1), temp(0, 2),
						temp(1, 0), temp(1, 1), temp(1, 2),
						temp(2, 0), temp(2, 1), temp(2, 2)
					);

					result += (temp3.inverse() * N) * weight;
					add += weight;
				}
			}

			if (j0 || j1 || j2)
			{
				initialNormal[vId][0] = result[0];
				initialNormal[vId][1] = result[1];
				initialNormal[vId][2] = result[2];
			}

			initialNormal[vId] = initialNormal[vId].normalize();
			if (pId == 6737) 
			{
				printf("N: %f,%f,%f\n", initialNormal[vId][0], initialNormal[vId][1], initialNormal[vId][2]);
			}
		}
	}

	void HierarchicalScene::getVerticesNormalInBindPose(
		DArray<Vec3f>& initialNormal,
		DArray<Mat4f>& joint_inverseBindMatrix,
		DArray<Mat4f>& WorldMatrix,

		DArrayList<int>& point2Vertice,
		DArray<Vec4f>& bind_joints_0,
		DArray<Vec4f>& bind_joints_1,
		DArray<Vec4f>& bind_joints_2,
		DArray<Vec4f>& weights_0,
		DArray<Vec4f>& weights_1,
		DArray<Vec4f>& weights_2,

		Vec2u range
	) 
	{
		cuExecute(initialNormal.size(),
			GetVerticesNormalInBindPose,
			initialNormal,
			joint_inverseBindMatrix,
			WorldMatrix,

			point2Vertice,
			bind_joints_0,
			bind_joints_1,
			bind_joints_2,
			weights_0,
			weights_1,
			weights_2,

			range
		);
	}

	void HierarchicalScene::updatePoint2Vertice(DArrayList<int>& d_p2v,DArray<int>& d_v2p)
	{
		int num = 0;
		int verticeNum = 0;

		for (auto it : this->getMeshes())
		{
			num += it->points.size();
			verticeNum += it->vertices.size();
		}

		std::vector<int> v2p(verticeNum);
		CArray<uint> instanceNum(num); 
		
		int pIdOffset = 0;
		for (auto it : this->getMeshes())
		{	
			for (auto p2v : it->pointId_verticeId)
			{
				int pId = p2v.first + pIdOffset;
				instanceNum[pId] = p2v.second.size();					
			}
			pIdOffset += it->points.size();
		}

		CArrayList<int> cArrayList;
		cArrayList.resize(instanceNum);

		pIdOffset = 0;
		int vIdOffset = 0;
		for (auto it : this->getMeshes())
		{
			for (auto p2v : it->pointId_verticeId)
			{
				int pId = p2v.first + pIdOffset;

				auto& list = cArrayList[pId];
				for (size_t k = 0; k < p2v.second.size(); k++)
				{
					int vId = p2v.second[k] + vIdOffset;
					list.insert(vId);
					v2p[vId] = pId;
				}
			}
			pIdOffset += it->points.size();
			vIdOffset += it->vertices.size();
		}
		d_p2v.assign(cArrayList);
		d_v2p.assign(v2p);
	}


	ModelObject:: ~ModelObject() 
	{
		child.clear();
		parent.clear();

	};

	bool ModelObject:: operator==(const ModelObject& model)
	{
		return name == model.name && id == model.id;
	}

	HierarchicalScene::HierarchicalScene()
	{
		mTimeStart = -1;
		mTimeEnd = -1;

		mJointData = std::make_shared<JointInfo>();
		mSkinData = std::make_shared <SkinInfo>() ;
		mJointAnimationData = std::make_shared<JointAnimationInfo>();
	}
	HierarchicalScene::~HierarchicalScene() { clear(); }

	void HierarchicalScene::clear()
	{
		mModelObjects.clear();
		mMeshes.clear();
		mBones.clear();
		mBoneRotations.clear();
		mBoneTranslations.clear();
		mBoneScales.clear();
		mBoneWorldMatrix.clear();
		mBoneInverseBindMatrix.clear();

		mJointData->clear();
		mJointAnimationData->clear();
		mSkinData->clear();

		mTimeStart = -1;
		mTimeEnd = -1;
	}

	int HierarchicalScene::findMeshIndexByName(std::string name)
	{
		int id = 0;
		for (auto it : mMeshes) {
			if (it->name == name) 
			{
				return id;
			}
			else
			{
				id++;
			}
		}
		std::cout << name << " not found !! "<<"\n";
		return -1;
	}

	int HierarchicalScene::minMeshIndex() 
	{
		return mMeshes[0]->id;
	}

	int HierarchicalScene::findObjectIndexByName(std::string name)
	{
		int id = 0;
		for (auto it : mModelObjects) {
			if (it->name == name)
				return id;

			id++;
		}
		return -1;
	}

	std::shared_ptr<ModelObject> HierarchicalScene::getObjectByName(std::string name)
	{
		for (auto it : mModelObjects) {
			if (it->name == name)
			{
				return it;
			}
		}

		return nullptr;
	}

	int HierarchicalScene::getObjIndexByName(std::string name)
	{
		int id = 0;
		for (auto it : mModelObjects) {
			if (it->name == name)
				return id;

			id++;
		}
		return -1;
	}

	int HierarchicalScene::getBoneIndexByName(std::string name)
	{
		int id = 0;
		for (auto it : mBones) {
			if (it->name == name)
				return id;

			id++;
		}
		return -1;
	}


	void HierarchicalScene::updateInverseBindMatrix()
	{
		std::vector<Mat4f> bindPoseLocalTransform;
		bindPoseLocalTransform.resize(mBones.size());
		for (auto it : mBones)
		{
			auto preR = it->preRotation;
			auto T = it->localTranslation;
			auto S = it->localScale;

			Quat<Real> pre =
				Quat<Real>(Real(M_PI) * preR[2] / 180, Vec3f(0, 0, 1))
				* Quat<Real>(Real(M_PI) * preR[1] / 180, Vec3f(0, 1, 0))
				* Quat<Real>(Real(M_PI) * preR[0] / 180, Vec3f(1, 0, 0));
			pre.normalize();
			Mat4f preMatrix = pre.toMatrix4x4();
			preMatrix = Mat4f(
				preMatrix(0,0) * S[0], preMatrix(0, 1), preMatrix(0, 2), T[0],
				preMatrix(1, 0), preMatrix(1, 1) * S[1], preMatrix(1, 2), T[1],
				preMatrix(2, 0), preMatrix(2, 1), preMatrix(2, 2) * S[2], T[2],
				preMatrix(3, 0), preMatrix(3, 1), preMatrix(3, 2), 1
			);
			bindPoseLocalTransform[it->id] = preMatrix;
		}

		if (mBoneInverseBindMatrix.size() != mBones.size())
			mBoneInverseBindMatrix.resize(mBones.size());

		for (auto it : mBones)
		{
			//printf("==================================================\n");
			int select = getBoneIndexByName(it->name);

			if (it->inverseBindMatrix == Mat4f::identityMatrix()) 
			{
				if (select == -1)continue;
				Mat4f inverseMatrix = bindPoseLocalTransform[it->id].inverse();
				for (size_t i = 0; i < it->parent.size(); i++)
				{
					auto parent = it->parent[i];
					inverseMatrix *= bindPoseLocalTransform[parent->id].inverse();
					//std::cout << parent->name << "\n";
					//coutMatrix(i, inverseMatrix);
				};

				mBoneInverseBindMatrix[select] = inverseMatrix;		
			}
			else
			{
				mBoneInverseBindMatrix[select] = it->inverseBindMatrix;
			}
			
		}
	}

	void HierarchicalScene::updateWorldTransformByKeyFrame(Real time) // ±º‰≤Â÷µ
	{

		//update Animation to mBoneRotations/mBoneTranslations/mBoneScales
		for (size_t i = 0; i < mBones.size(); i++)
		{
			int select = getBoneIndexByName(mBones[i]->name);
			if (select == -1)continue;

			//Rotation
			mBoneRotations[select].x = getVectorDataByTime(mJointAnimationData->mJoint_KeyId_R_X[select], mJointAnimationData->mJoint_KeyId_tR_X[select], time);
			mBoneRotations[select].y = getVectorDataByTime(mJointAnimationData->mJoint_KeyId_R_Y[select], mJointAnimationData->mJoint_KeyId_tR_Y[select], time);
			mBoneRotations[select].z = getVectorDataByTime(mJointAnimationData->mJoint_KeyId_R_Z[select], mJointAnimationData->mJoint_KeyId_tR_Z[select], time);
			//Translation
			mBoneTranslations[select].x = getVectorDataByTime(mJointAnimationData->mJoint_KeyId_T_X[select], mJointAnimationData->mJoint_KeyId_tT_X[select], time);
			mBoneTranslations[select].y = getVectorDataByTime(mJointAnimationData->mJoint_KeyId_T_Y[select], mJointAnimationData->mJoint_KeyId_tT_Y[select], time);
			mBoneTranslations[select].z = getVectorDataByTime(mJointAnimationData->mJoint_KeyId_T_Z[select], mJointAnimationData->mJoint_KeyId_tT_Z[select], time);
			//Scale
			mBoneScales[select].x = getVectorDataByTime(mJointAnimationData->mJoint_KeyId_S_X[select], mJointAnimationData->mJoint_KeyId_tS_X[select], time);
			mBoneScales[select].y = getVectorDataByTime(mJointAnimationData->mJoint_KeyId_S_Y[select], mJointAnimationData->mJoint_KeyId_tS_Y[select], time);
			mBoneScales[select].z = getVectorDataByTime(mJointAnimationData->mJoint_KeyId_S_Z[select], mJointAnimationData->mJoint_KeyId_tS_Z[select], time);
		}

		for (auto it : mBones)
		{
			int select = getBoneIndexByName(it->name);
			if (select == -1)continue;

			Mat4f worldMatrix = it->localTransform;
			for (size_t i = 0; i < it->parent.size(); i++) {
				auto parent = it->parent[i];
				worldMatrix *= parent->localTransform;
			};

			mBoneWorldMatrix[select] = worldMatrix;
		}
		currentTime = time;

	};


	void HierarchicalScene::updateBoneWorldMatrix()
	{
		if (mBoneWorldMatrix.size() != mBones.size())
			mBoneWorldMatrix.resize(mBones.size());

		for (auto it : mBones)
		{
			
			int select = getBoneIndexByName(it->name);
			if (select == -1)continue;

			{
				Mat4f worldMatrix = it->localTransform;

				for (size_t i = 0; i < it->parent.size(); i++) {
					auto parent = it->parent[i];
					worldMatrix = parent->localTransform * worldMatrix;
				};

				mBoneWorldMatrix[select] = worldMatrix;
				it->worldTransform = worldMatrix;
			}
		}

	}

	void HierarchicalScene::updateMeshWorldMatrix()
	{
		for (auto it : mMeshes)
		{

			int select = getObjIndexByName(it->name);
			if (select == -1)continue;

			{
				Mat4f worldMatrix = it->localTransform;

				for (size_t i = 0; i < it->parent.size(); i++) {
					auto parent = it->parent[i];
					worldMatrix = parent->localTransform * worldMatrix;
				};
				it->worldTransform = worldMatrix;
			}
		}
	}

	Real HierarchicalScene::getVectorDataByTime(std::vector<Real> data, std::vector<Real> timeCode, Real time)
	{
		if (!bool(data.size()))
			return 0;

		int idx = findMaxSmallerIndex(timeCode, time);
		if (idx >= data.size() - 1) {				//   [size-1]<=[tId]  
			return data[data.size() - 1];
		}
		else if (idx >= 0) {
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

	int HierarchicalScene::findMaxSmallerIndex(const std::vector<float>& arr, float v) {
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

	void HierarchicalScene::pushBackBone(std::shared_ptr<Bone> bone)
	{
		mModelObjects.push_back(bone);
		mBones.push_back(bone);
		bone->id = mModelObjects.size() - 1;
		createLocalTransform(bone);
	}
	void HierarchicalScene::pushBackMesh(std::shared_ptr<MeshInfo> mesh)
	{
		mModelObjects.push_back(mesh);
		mMeshes.push_back(mesh);
		mesh->id = mModelObjects.size() - 1;
		createLocalTransform(mesh);

	}

	MeshInfo::MeshInfo() {};

	MeshInfo::~MeshInfo()
	{
		vertices.clear();
		verticeId_pointId.clear();
		pointId_verticeId.clear();
		normals.clear();
		texcoords.clear();
		verticesColor.clear();
		facegroup_triangles.clear();
		facegroup_polygons.clear();
		materials.clear();
		boundingBox.clear();
		boundingTransform.clear();
		boneIndices0.clear();
		boneWeights0.clear();
		boneIndices1.clear();
		boneWeights1.clear();
		boneIndices2.clear();
		boneWeights2.clear();
	}

	void MeshInfo::resizeSkin(int size)
	{
		boneIndices0.resize(size);
		boneWeights0.resize(size);
		boneIndices1.resize(size);
		boneWeights1.resize(size);
		boneIndices2.resize(size);
		boneWeights2.resize(size);
	}

	void HierarchicalScene::UpdateJointData()
	{
		this->updateInverseBindMatrix();
		this->updateBoneWorldMatrix();

		std::vector<int> JointsId;
		std::map<int, std::vector<int>> jointDir;
		std::map<int, Vec3f> BindPoseTranslation;
		std::map<int, Vec3f> BindPoseScale;
		std::map<int, Vec3f> BindPoseRotator;
		std::map<int, Vec3f> BindPosePreTranslation;
		std::map<int, Vec3f> BindPosePreScale;
		std::map<int, Vec3f> BindPosePreRotator;
		std::vector<Mat4f> localMatrix;

		for (auto it : mBones)
		{
			BindPoseTranslation[it->id] = it->localTranslation;
			BindPoseScale[it->id] = it->localScale;
			BindPosePreRotator[it->id] = it->preRotation;

			BindPoseRotator[it->id] = it->localRotation;

			JointsId.push_back(it->id);
			std::shared_ptr<Bone> currentJoint = it;

			jointDir[it->id].push_back(it->id);

			for (size_t i = 0; i < it->parent.size(); i++) {
				jointDir[it->id].insert(jointDir[it->id].begin(), it->parent[i]->id);
			};
		}

		int maxBoneId = *(std::max_element(JointsId.begin(), JointsId.end()));
		localMatrix.resize(maxBoneId + 1);

		std::map<int,std::string> boneName;
		for (auto it : mBones)
		{
			localMatrix[it->id] = it->localTransform;
			boneName[it->id] = it->name;
		}

		mJointData->setJointName(boneName);
		mJointData->SetJointInfo(mBoneInverseBindMatrix, localMatrix, mBoneWorldMatrix, JointsId, jointDir, BindPoseTranslation, BindPoseScale, BindPoseRotator, BindPosePreRotator);
		
		mJointAnimationData->mSkeleton = mJointData;
		mJointAnimationData->resizeJointsData(maxBoneId + 1);

	};

	void HierarchicalScene::coutBoneHierarchial()
	{

		for (auto it : mBones)
		{
			printf("%s :", it->name.c_str());

			for (size_t i = 0; i < it->parent.size(); i++) {
				printf("- %s ", it->parent[i]->name.c_str());

			};
			printf("\n");

		}
	}


	void HierarchicalScene::updateSkinData(std::shared_ptr<TextureMesh> texMesh)
	{

		int tempSize = 0;
		for (int meshId = 0; meshId < mMeshes.size(); meshId++)
		{
			auto mesh = mMeshes[meshId];
			mSkinData->pushBack_Data(mesh->boneWeights0, mesh->boneWeights1, mesh->boneIndices0, mesh->boneIndices1, &mesh->boneWeights2, &mesh->boneIndices2);

			mSkinData->skin_VerticeRange[meshId] = Vec2u(tempSize, tempSize + mesh->points.size());
			tempSize += mesh->points.size();
		}

		mSkinData->mesh = texMesh;
		mSkinData->initialPosition.assign(texMesh->vertices());
		mSkinData->initialNormal.assign(texMesh->normals());

	}

	Mat4f HierarchicalScene::createLocalTransform(std::shared_ptr<ModelObject> object) {
		//The FBX file's joints have preRotation values during construction, and when the user operates the LocalRotation values, the actual rotation is Quat(preRotation) * Quat(LocalRotation).
		auto& R = object->localRotation;
		auto& preR = object->preRotation;
		auto& S = object->localScale;
		auto& T = object->localTranslation;

		Quat<Real> rot =
			Quat<Real>(Real(M_PI) * R[2] / 180, Vec3f(0, 0, 1))
			* Quat<Real>(Real(M_PI) * R[1] / 180, Vec3f(0, 1, 0))
			* Quat<Real>(Real(M_PI) * R[0] / 180, Vec3f(1, 0, 0));
		rot.normalize();

		Quat<Real> pre =
			Quat<Real>(Real(M_PI) * preR[2] / 180, Vec3f(0, 0, 1))
			* Quat<Real>(Real(M_PI) * preR[1] / 180, Vec3f(0, 1, 0))
			* Quat<Real>(Real(M_PI) * preR[0] / 180, Vec3f(1, 0, 0));
		pre.normalize();

		auto q = pre * rot;

		Mat4f preMatrix = q.toMatrix4x4();

		Mat4f m = Mat4f(preMatrix(0, 0) * S.x, preMatrix(0, 1), preMatrix(0, 2), T.x,
			preMatrix(1, 0), preMatrix(1, 1) * S.y, preMatrix(1, 2), T.y,
			preMatrix(2, 0), preMatrix(2, 1), preMatrix(2, 2) * S.z, T.z,
			preMatrix(3, 0), preMatrix(3, 1), preMatrix(3, 2), preMatrix(3, 3));

		object->localTransform = m;
		//std::cout << object->name << " Initial R :\n "
		//	<< R.x << ", " << R.y << ", " << R.z << ", \n"
		//	<< preR.x << ", " << preR.y << ", " << preR.z << ", \n"
		//	<< q.x << ", " << q.y << ", " << q.z << ", " << q.w << ", \n";
		return m;

	}

	Real HierarchicalScene::lerp(Real v0, Real v1, float weight)
	{
		return v0 + (v1 - v0) * weight;
	}

	void HierarchicalScene::showJointInfo()
	{
		std::string str;

		for (auto it : mBones)
		{
			if (!it->parent.size())
			{
				str.append(it->name);
				buildTree(str, it->child, 1);
			}
		}
		std::cout << str << "\n";
	}


	template< typename Coord, typename Mat4f, typename Vec3f >
	__global__ void ShapeTransform(
		DArray<Coord> intialPosition,
		DArray<Vec3f> worldPosition,
		DArray<Coord> intialNormal,
		DArray<Vec3f> Normal,
		DArray<Mat4f> WorldMatrix,
		DArray<uint> vertexId_shape,
		DArray<int> shapeId_MeshId
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intialPosition.size()) return;

		int shape = vertexId_shape[pId];

		int MeshId = shapeId_MeshId[shape];

		Vec4f tempV = Vec4f(intialPosition[pId][0], intialPosition[pId][1], intialPosition[pId][2], 1);
		Vec4f tempN = Vec4f(intialNormal[pId][0], intialNormal[pId][1], intialNormal[pId][2], 0);
		if (pId == 1)
		{
			auto iP = intialPosition[pId];
		}

		tempV = WorldMatrix[MeshId] * tempV;
		tempN = WorldMatrix[MeshId] * tempN;

		worldPosition[pId] = Coord(tempV[0], tempV[1], tempV[2]);
		Normal[pId] = Coord(tempN[0], tempN[1], tempN[2]);

	}


	template< typename Vec3f, typename Mat4f>
	void HierarchicalScene::shapeTransform(
		DArray<Vec3f>& intialPosition,
		DArray<Vec3f>& worldPosition,
		DArray<Vec3f>& intialNormal,
		DArray<Vec3f>& Normal,
		DArray<Mat4f>& WorldMatrix,
		DArray<uint>& vertexId_shape,
		DArray<int>& shapeId_MeshId
	) 
	{
		cuExecute(intialPosition.size(),
			ShapeTransform,
			intialPosition,
			worldPosition,
			intialNormal,
			Normal,
			WorldMatrix,
			vertexId_shape,
			shapeId_MeshId
		);
	}

	template void HierarchicalScene::shapeTransform <Vec3f, Mat4f>(
		DArray<Vec3f>& intialPosition,
		DArray<Vec3f>& worldPosition,
		DArray<Vec3f>& intialNormal,
		DArray<Vec3f>& Normal,
		DArray<Mat4f>& WorldMatrix,
		DArray<uint>& vertexId_shape,
		DArray<int>& shapeId_MeshId
		);

	template< typename Vec3f, typename Mat4f >
	__global__ void TextureMeshTransform(
		DArray<Vec3f> intialPosition,
		DArray<Vec3f> worldPosition,
		DArray<Vec3f> intialNormal,
		DArray<Vec3f> Normal,
		Mat4f WorldMatrix
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intialPosition.size()) return;

		Vec4f tempV = Vec4f(intialPosition[pId][0], intialPosition[pId][1], intialPosition[pId][2], 1);
		Vec4f tempN = Vec4f(intialNormal[pId][0], intialNormal[pId][1], intialNormal[pId][2], 0);
		if (pId == 1)
		{
			auto iP = intialPosition[pId];
		}

		tempV = WorldMatrix * tempV;
		tempN = WorldMatrix * tempN;

		worldPosition[pId] = Vec3f(tempV[0], tempV[1], tempV[2]);
		Normal[pId] = Vec3f(tempN[0], tempN[1], tempN[2]);


	}

	template< typename Vec3f, typename Mat4f>
	void HierarchicalScene::textureMeshTransform(
		DArray<Vec3f>& intialPosition,
		DArray<Vec3f>& worldPosition,
		DArray<Vec3f>& intialNormal,
		DArray<Vec3f>& Normal,
		Mat4f& WorldMatrix
	) 
	{
		cuExecute(intialPosition.size(),
			TextureMeshTransform,
			intialPosition,
			worldPosition,
			intialNormal,
			Normal,
			WorldMatrix
		);
	}

	template void HierarchicalScene::textureMeshTransform <Vec3f, Mat4f>(DArray<Vec3f>& intialPosition,
		DArray<Vec3f>& worldPosition,
		DArray<Vec3f>& intialNormal,
		DArray<Vec3f>& Normal,
		Mat4f& WorldMatrix
		);


	template< typename Vec3f, typename uint>
	__global__ void ShapeToCenter(
		DArray<Vec3f> iniPos,
		DArray<Vec3f> finalPos,
		DArray<uint> shapeId,
		DArray<Vec3f> t
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= iniPos.size()) return;

		finalPos[pId] = iniPos[pId] - t[shapeId[pId]];
		Vec4f P = Vec4f(finalPos[pId][0], finalPos[pId][1], finalPos[pId][2], 1);

		finalPos[pId] = Vec3f(P[0], P[1], P[2]);

	}
	template< typename Vec3f, typename uint>
	void HierarchicalScene::shapeToCenter(DArray<Vec3f>& iniPos,
		DArray<Vec3f>& finalPos,
		DArray<uint>& shapeId,
		DArray<Vec3f>& t
	) 
	{
		cuExecute(iniPos.size(),
			ShapeToCenter,
			iniPos,
			finalPos,
			shapeId,
			t
		);
	
	}

	template void HierarchicalScene::shapeToCenter <Vec3f, uint>(DArray<Vec3f>& iniPos,
		DArray<Vec3f>& finalPos,
		DArray<uint>& shapeId,
		DArray<Vec3f>& t
		);


	template< typename Triangle, typename Vec3f >
	__global__ void computeTriangleNormal(
		DArray<Triangle> triangle,
		DArray<Vec3f> pos,
		DArray<Vec3f> triangleNormal
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangle.size()) return;

		int pA = triangle[tId][0];
		int pB = triangle[tId][1];
		int pC = triangle[tId][2];
		Vec3f AB = pos[pB] - pos[pA];
		Vec3f BC = pos[pC] - pos[pB];
		AB = AB.normalize();
		BC = BC.normalize();

		triangleNormal[tId] = BC.cross(AB);
		triangleNormal[tId] = triangleNormal[tId].normalize();
	}

	template< typename Triangle, typename Vec3f >
	__global__ void computeVerticesNormal(
		DArray<Triangle> normalIndex,
		DArray<int> ver2Point,
		DArrayList<int> point2Triangle,
		DArray<Vec3f> triangleNormal,
		DArray<Vec3f> verticesNormal
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= normalIndex.size()) return;

		for (int i = 0; i < 3; i++)
		{
			int vId = normalIndex[tId][i];
			int pId = ver2Point[vId];

			Vec3f vN = Vec3f(0);
			for (int j = 0; j < point2Triangle[pId].size(); j++)
			{
				int triId = point2Triangle[pId][j];

				vN += triangleNormal[triId];
			}

			verticesNormal[vId] = vN.normalize();
		}

	}

	__global__ void initialV2P(
		DArray<int> ver2Point
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= ver2Point.size()) return;

		ver2Point[tId] = tId;
	}


	void HierarchicalScene::computeTexMeshVerticesNormal(
		std::vector<std::shared_ptr<Shape>>& shapes,
		DArray<Vec3f>& Position,
		DArray<Vec3f>& Normal,
		DArray<int>* vertices2Point)
	{
		for (auto shape : shapes)
		{
			auto triSetHelper = std::make_shared<TriangleSet<DataType3f>>();
			triSetHelper->setPoints(Position);
			triSetHelper->setTriangles(shape->vertexIndex);
			triSetHelper->update();

			DArray<Vec3f> triNormal;
			triNormal.resize(shape->vertexIndex.size());


			const DArrayList<int>& point2Tri = triSetHelper->vertex2Triangle();

			cuExecute(shape->vertexIndex.size(),
				computeTriangleNormal,
				shape->vertexIndex,
				Position,
				triNormal
			);

			DArray<int> d_v2p;
			DArray<int>& v2p = *vertices2Point;
			if (!vertices2Point)
			{
				
				d_v2p.resize(Position.size());

				cuExecute(d_v2p.size(),
					initialV2P,
					d_v2p
				);
				v2p = d_v2p;
			}

			cuExecute(shape->normalIndex.size(),
				computeVerticesNormal,
				shape->normalIndex,
				v2p,
				point2Tri,
				triNormal,
				Normal
			);

			triNormal.clear();
			d_v2p.clear();
		}

	}

	template< typename Vec3f >
	__global__ void flipNormalData(
		DArray<Vec3f> normal
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= normal.size()) return;

		normal[tId] = normal[tId] * -1;
	}

	void HierarchicalScene::flipNormal(DArray<Vec3f>& Normal) 
	{
		cuExecute(Normal.size(),
			flipNormalData,
			Normal
		);
	}

	
}