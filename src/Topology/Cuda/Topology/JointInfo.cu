
#include "JointInfo.h"
#include "Matrix.h"

namespace dyno
{
	IMPLEMENT_CLASS(JointInfo)
	std::vector<int> getJointDirByJointIndex(int Index, std::map<int, std::vector<int>> jointId_joint_Dir)
	{
		std::vector<int> jointDir;
		std::map<int, std::vector<int>>::const_iterator iter;

		//get skeletal chain
		iter = jointId_joint_Dir.find(Index);
		if (iter == jointId_joint_Dir.end())
		{
			std::cout << "Error: not found JointIndex \n";
			return jointDir;
		}

		jointDir = iter->second;
		return jointDir;
	}


	template< typename Vec3f, typename Quat1f ,typename Mat4f>
	__global__ void updateLocalMatrix(
		DArray<Vec3f> translation,
		DArray<Vec3f> scale,
		DArray<Quat1f> rotation,
		DArray<Mat4f> localMatrix,
		DArray<int> jointIds
	) 
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= jointIds.size()) return;

		int joint = jointIds[pId];

		Mat4f r = rotation[joint].toMatrix4x4();
		Mat4f s = Mat4f
					(scale[joint][0], 0, 0, 0,
						0, scale[joint][1], 0, 0,
						0, 0, scale[joint][2], 0,
						0, 0, 0, 1
					);
		Mat4f t = Mat4f
					(1, 0, 0, translation[joint][0],
					0, 1, 0, translation[joint][1],
					0, 0, 1, translation[joint][2],
					0, 0, 0, 1
					);
		localMatrix[joint] = t * s * r;

		Mat4f c = localMatrix[joint];
	

	}



	void JointInfo::updateWorldMatrixByTransform()
	{
		DArray<int> jointIds;
		jointIds.assign(mAllJoints);

		mJointLocalMatrix = getLocalMatrix(this->currentPose);


		std::vector<Mat4f> c_joint_Mat4f;
		c_joint_Mat4f.resize(mMaxJointID + 1);


		for (size_t i = 0; i < mAllJoints.size(); i++)
		{
			joint jointId = mAllJoints[i];
			const std::vector<int>& jD = getJointDirByJointIndex(jointId, mJointDir);

			//printf("AAA  %d : ", jointId);//
			Mat4f tempMatrix = Mat4f::identityMatrix();

			//
			for (int k = jD.size() - 1; k >= 0; k--)
			{
				joint select = jD[k];
				//printf("-  %d : ", select);//
				if(useLeftHandedCoordSystem)
					tempMatrix = mJointLocalMatrix[select] * tempMatrix;
				else
					tempMatrix *= mJointLocalMatrix[select];

			}
			//printf("\n");
			c_joint_Mat4f[jointId] = tempMatrix;

		}

		//printf("********** jointAnimation : %d  ***********\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n ***********************\nQuat : %f,%f,%f,%f\nScale : %f,%f,%f\nTrans : %f,%f,%f\n***********************\n",
		//	joint,
		//	c(0, 0), c(0, 1), c(0, 2), c(0, 3),
		//	c(1, 0), c(1, 1), c(1, 2), c(1, 3),
		//	c(2, 0), c(2, 1), c(2, 2), c(2, 3),
		//	c(3, 0), c(3, 1), c(3, 2), c(3, 3),

		//	rotation[joint].x, rotation[joint].y, rotation[joint].z, rotation[joint].w,
		//	scale[joint][0], scale[joint][1], scale[joint][2],
		//	translation[joint][0], translation[joint][1], translation[joint][2]
		//);


		mJointWorldMatrix.assign(c_joint_Mat4f);


	}




	void JointInfo::setJoint(const JointInfo& j)
	{
		mJointInverseBindMatrix.assign(j.mJointInverseBindMatrix);
		mJointLocalMatrix=(j.mJointLocalMatrix);
		mJointWorldMatrix.assign(j.mJointWorldMatrix);
		mAllJoints = j.mAllJoints;
		mJointDir = j.mJointDir;
		mMaxJointID = j.mMaxJointID;
		mBindPoseTranslation = j.mBindPoseTranslation;
		mBindPoseScale = j.mBindPoseScale;
		mBindPoseRotation = j.mBindPoseRotation;

		mJointName = j.mJointName;
	}

	bool JointInfo::isEmpty()
	{
		if (mJointInverseBindMatrix.isEmpty() || mJointLocalMatrix.empty() || mJointWorldMatrix.isEmpty())
			return true;
	}

	void JointInfo::setGltfJointInfo(
		DArray<Mat4f>& InverseBindMatrix,
		std::vector<Mat4f>& LocalMatrix,
		DArray<Mat4f>& WorldMatrix,
		std::vector<int>& allJoints,
		std::map<joint, std::vector<joint>>& jointDir,
		std::map<joint, Vec3f>& bindPoseTranslation,
		std::map<joint, Vec3f>& bindPoseScale,
		std::map<joint, Quat1f>& bindPoseRotation
	)
	{
		this->useLeftHandedCoordSystem = false;
		mJointInverseBindMatrix.assign(InverseBindMatrix);

		mJointLocalMatrix = LocalMatrix;
		
		mJointWorldMatrix.assign(WorldMatrix);
		mAllJoints = allJoints;
		mJointDir = jointDir;
		if (mAllJoints.size())
			mMaxJointID = *(std::max_element(allJoints.begin(), allJoints.end()));

		std::vector<Vec3f> tempT;
		std::vector<Vec3f> tempS;
		std::vector<Quat1f> tempR;

		mBindPoseTranslation.resize(mMaxJointID + 1);
		mBindPoseScale.resize(mMaxJointID + 1);
		mBindPoseRotation.resize(mMaxJointID + 1);

		for (auto it : mAllJoints)
		{
			auto iterT = bindPoseTranslation.find(it);
			if (iterT != bindPoseTranslation.end())
				mBindPoseTranslation[it] = bindPoseTranslation[it];
			else
				mBindPoseTranslation[it] = Vec3f(0.0f);

			auto iterS = bindPoseScale.find(it);
			if (iterS != bindPoseScale.end())
				mBindPoseScale[it] = bindPoseScale[it];
			else
				mBindPoseScale[it] = Vec3f(1.0f);

			auto iterR = bindPoseRotation.find(it);
			if (iterR != bindPoseRotation.end())
				mBindPoseRotation[it] = bindPoseRotation[it];
			else
				mBindPoseRotation[it] = Quat1f();
		}

	}


	JointInfo::~JointInfo()
	{
		this->clear();
		mJointInverseBindMatrix.clear();
		mJointLocalMatrix.clear();
		mJointWorldMatrix.clear();


	};

	//JointInfo::JointInfo(
	//	DArray<Mat4f>& InverseBindMatrix,
	//	DArray<Mat4f>& LocalMatrix,
	//	DArray<Mat4f>& WorldMatrix,
	//	std::vector<int>& allJoints,
	//	std::map<joint, std::vector<joint>>& jointDir,
	//	std::map<joint, Vec3f>& bindPoseTranslation,
	//	std::map<joint, Vec3f>& bindPoseScale,
	//	std::map<joint, Quat1f>& bindPoseRotation
	//) 
	//{
	//	setGltfJointInfo(InverseBindMatrix,
	//		LocalMatrix,
	//		WorldMatrix,
	//		allJoints,
	//		jointDir,
	//		bindPoseTranslation,
	//		bindPoseScale,
	//		bindPoseRotation
	//	);
	//}


	float JointAnimationInfo::calculateMinTime(const std::map<joint, std::vector<Real>>& timeCodes) {
		float minTime = std::numeric_limits<float>::max();
		for (const auto& it : timeCodes) {
			if (!it.second.empty()) {
				float tempMin = *std::min_element(it.second.begin(), it.second.end());
				minTime = std::min(minTime, tempMin);
			}
		}
		return minTime;
	}

	float JointAnimationInfo::calculateMaxTime(const std::map<joint, std::vector<Real>>& timeCodes) {
		float maxTime = std::numeric_limits<float>::lowest();
		for (const auto& it : timeCodes) {
			if (!it.second.empty()) {
				float tempMax = *std::max_element(it.second.begin(), it.second.end());
				maxTime = std::max(maxTime, tempMax);
			}
		}
		return maxTime;
	}



	void JointAnimationInfo::setGLTFAnimationData(
		std::map<joint, std::vector<Vec3f>>& jointTranslation,
		std::map<joint, std::vector<Real>>& jointTimeCodeTranslation,
		std::map<joint, std::vector<Vec3f>>& jointScale,
		std::map<joint, std::vector<Real>>& jointIndexTimeCodeScale,
		std::map<joint, std::vector<Quat1f>>& jointRotation,
		std::map<joint, std::vector<Real>>& jointIndexRotation,
		std::shared_ptr<JointInfo> skeleton,
		bool loop 
	)//GLTF
	{
		this->clear();
		this->setLoop(loop);

		mJoint_KeyId_QuatRotation = jointRotation;
		mJoint_KeyId_tQuatRotation = jointIndexRotation;

		//--------------------------------------------
		mSkeleton = skeleton;

		resizeJointsData(mSkeleton->mMaxJointID + 1);

		for (size_t i = 0; i < jointTranslation.size(); i++)
		{
			auto Jid_T = jointTranslation[i];
			auto Jid_tT = jointTimeCodeTranslation[i];
			for (auto T : Jid_T)
			{
				this->mJoint_KeyId_T_X[i].push_back(T[0]);
				this->mJoint_KeyId_T_Y[i].push_back(T[1]);
				this->mJoint_KeyId_T_Z[i].push_back(T[2]);
			}
			for (auto tT : Jid_tT)
			{
				this->mJoint_KeyId_tT_X[i].push_back(tT);
				this->mJoint_KeyId_tT_Y[i].push_back(tT);
				this->mJoint_KeyId_tT_Z[i].push_back(tT);
			}
		}

		mJoint_KeyId_QuatRotation = jointRotation;
		mJoint_KeyId_tQuatRotation = jointIndexRotation;


		for (size_t i = 0; i < jointScale.size(); i++)
		{
			auto Jid_S = jointScale[i];
			auto Jid_tS = jointIndexTimeCodeScale[i];
			for (auto S : Jid_S)
			{
				this->mJoint_KeyId_S_X[i].push_back(S[0]);
				this->mJoint_KeyId_S_Y[i].push_back(S[1]);
				this->mJoint_KeyId_S_Z[i].push_back(S[2]);
			}
			for (auto tS : Jid_tS)
			{
				this->mJoint_KeyId_tS_X[i].push_back(tS);
				this->mJoint_KeyId_tS_Y[i].push_back(tS);
				this->mJoint_KeyId_tS_Z[i].push_back(tS);
			}
		}
		updateTotalTime();
	};

	void JointAnimationInfo::updateTotalTime() 
	{
		float startGltfR;
		float endGltfR;
		float startRx;
		float startRy;
		float startRz;
		float endRx;
		float endRy;
		float endRz;

		if (isGltfAnimation())
		{
			startGltfR = calculateMinTime(mJoint_KeyId_tQuatRotation);
			endGltfR = calculateMaxTime(mJoint_KeyId_tQuatRotation);
		}
		else 
		{
			startRx = calculateMinTime(mJoint_KeyId_tR_X);
			startRy = calculateMinTime(mJoint_KeyId_tR_Y);
			startRz = calculateMinTime(mJoint_KeyId_tR_Z);
			endRx = calculateMaxTime(mJoint_KeyId_tR_X);
			endRy = calculateMaxTime(mJoint_KeyId_tR_Y);
			endRz = calculateMaxTime(mJoint_KeyId_tR_Z);
		}


		float startTx = calculateMinTime(mJoint_KeyId_tT_X);
		float startTy = calculateMinTime(mJoint_KeyId_tT_Y);
		float startTz = calculateMinTime(mJoint_KeyId_tT_Z);
		float startSx = calculateMinTime(mJoint_KeyId_tS_X);
		float startSy = calculateMinTime(mJoint_KeyId_tS_Y);
		float startSz = calculateMinTime(mJoint_KeyId_tS_Z);


		float endTx = calculateMaxTime(mJoint_KeyId_tT_X);
		float endTy = calculateMaxTime(mJoint_KeyId_tT_Y);
		float endTz = calculateMaxTime(mJoint_KeyId_tT_Z);
		float endSx = calculateMaxTime(mJoint_KeyId_tS_X);
		float endSy = calculateMaxTime(mJoint_KeyId_tS_Y);
		float endSz = calculateMaxTime(mJoint_KeyId_tS_Z);

		float timeMin;
		float timeMax;
		if (isGltfAnimation()) 
		{
			timeMin = std::min({ startTx,startTy,startTz, startGltfR, startSx,startSy,startSz });
			timeMax = std::max({ endTx,endTy,endTz, endGltfR, endSx,endSy,endSz });
		}
		else 
		{
			timeMin = std::min({ startTx,startTy,startTz, startRz,startRy,startRz, startSx,startSy,startSz });
			timeMax = std::max({ endTx,endTy,endTz, endRx,endRy,endRz, endSx,endSy,endSz });
		}
		
		if (timeMin != 0) {
			offsetTimeCodes(mJoint_KeyId_tT_X, timeMin);
			offsetTimeCodes(mJoint_KeyId_tT_Y, timeMin);
			offsetTimeCodes(mJoint_KeyId_tT_Z, timeMin);

			offsetTimeCodes(mJoint_KeyId_tS_X, timeMin);
			offsetTimeCodes(mJoint_KeyId_tS_Y, timeMin);
			offsetTimeCodes(mJoint_KeyId_tS_Z, timeMin);

			offsetTimeCodes(mJoint_KeyId_tQuatRotation, timeMin);

		}
		mTotalTime = timeMax - timeMin;
	}

	void JointAnimationInfo::updateJointsTransform(float time)
	{
		if (currentTime == time * mPlayRate)
			return;

		currentTime = time * mPlayRate;


		if (mSkeleton != NULL) 
		{
			for (size_t i = 0; i < mSkeleton->mAllJoints.size(); i++)
			{
				joint select = mSkeleton->mAllJoints[i];
				updateTransform(select ,time);
			}	
		}

	}

	Transform3f JointAnimationInfo::updateTransform(joint select, float time) //时间插值
	{
		bool useGltf = bool(mJoint_KeyId_QuatRotation.size());
		if (useGltf) //GLTF
		{
			updateGLTFRotation(select,time);
		}
		else
		{
			updateCurrentPose(select, time, this->mRotator, mSkeleton->mBindPoseRotator, this->mJoint_KeyId_R_X, this->mJoint_KeyId_tR_X, 0);
			updateCurrentPose(select, time, this->mRotator, mSkeleton->mBindPoseRotator, this->mJoint_KeyId_R_Y, this->mJoint_KeyId_tR_Y, 1);
			updateCurrentPose(select, time, this->mRotator, mSkeleton->mBindPoseRotator, this->mJoint_KeyId_R_Z, this->mJoint_KeyId_tR_Z, 2);

			//The FBX file's joints have preRotation values during construction, and when the user operates the LocalRotation values, the actual rotation is Quat(preRotation) * Quat(LocalRotation).
			auto& R = this->mRotator[select];//this->mRotator[select];
			auto& preR = this->mSkeleton->mBindPosePreRotator[select];

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

			auto q = pre * rot  ;//

			mQuatRotation[select] = q;

			//std::cout << this->mSkeleton->mJointName[select] << " ani R :\n "
			//	<< R.x << ", " << R.y << ", " << R.z << ", \n"
			//	<< preR.x << ", " << preR.y << ", " << preR.z << ", \n"
			//	<< q.x << ", " << q.y << ", " << q.z << ", " << q.w << ", \n";
			
		}

		updateCurrentPose(select, time, this->mScale, mSkeleton->mBindPoseScale, this->mJoint_KeyId_S_X, this->mJoint_KeyId_tS_X, 0);
		updateCurrentPose(select, time, this->mScale, mSkeleton->mBindPoseScale, this->mJoint_KeyId_S_Y, this->mJoint_KeyId_tS_Y, 1);
		updateCurrentPose(select, time, this->mScale, mSkeleton->mBindPoseScale, this->mJoint_KeyId_S_Z, this->mJoint_KeyId_tS_Z, 2);

		updateCurrentPose(select, time, this->mTranslation, mSkeleton->mBindPoseTranslation, this->mJoint_KeyId_T_X, this->mJoint_KeyId_tT_X, 0);
		updateCurrentPose(select, time, this->mTranslation, mSkeleton->mBindPoseTranslation, this->mJoint_KeyId_T_Y, this->mJoint_KeyId_tT_Y, 1);
		updateCurrentPose(select, time, this->mTranslation, mSkeleton->mBindPoseTranslation, this->mJoint_KeyId_T_Z, this->mJoint_KeyId_tT_Z, 2);

		
		return Transform3f(mTranslation[select], mQuatRotation[select].toMatrix3x3(), mScale[select]);


	};

	std::vector<Vec3f> JointAnimationInfo::getJointsTranslation(float time)
	{
		updateJointsTransform(time);
		return mTranslation;
	}

	void JointInfo::SetJointInfo(
		std::vector<Mat4f>& InverseBindMatrix,
		std::vector<Mat4f>& LocalMatrix,
		std::vector<Mat4f>& WorldMatrix,
		std::vector<int>& allJoints,
		std::map<int, std::vector<int>>& jointDir,
		std::map<int, Vec3f>& bindPoseTranslation,
		std::map<int, Vec3f>& bindPoseScale,
		std::map<int, Vec3f>& bindPoseRotation,
		std::map<int, Vec3f>& bindPosePreRotation
	) 
	{
		//std::locale::global(std::locale("zh_CN.UTF-8"));

		//for (auto it : bindPoseTranslation)
		//{
		//	auto id = it.first;
		//	auto vec = it.second;
		//	std::cout << mJointName[id];
		//	printf("- %d : %f,%f,%f\n",id, vec[0],vec[1],vec[2]);
		//}
		//for (auto it : jointDir)
		//{
		//	std::cout << mJointName[it.first] << ": ";
		//	for (auto jj : it.second)
		//	{
		//		std::cout << " - " << mJointName[jj];
		//	}
		//	std::cout << std::endl;
		//}

		//for (auto it : allJoints)
		//{
		//	std::cout << mJointName[it] << ": \n";
		//	Mat4f temp = InverseBindMatrix[it];
		//	printf("%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n", 
		//		temp(0, 0), temp(0, 1), temp(0, 2), temp(0, 3), 
		//		temp(1, 0), temp(1, 1), temp(1, 2), temp(2, 3), 
		//		temp(2, 0), temp(2, 1), temp(2, 2), temp(2, 3), 
		//		temp(3, 0), temp(3, 1), temp(3, 2), temp(3, 3));
		//}

		mJointInverseBindMatrix.assign(InverseBindMatrix);
		
		mJointWorldMatrix.assign(WorldMatrix);
		mJointLocalMatrix = LocalMatrix;

		mAllJoints = allJoints;
		mJointDir = jointDir;
		if (mAllJoints.size())
			mMaxJointID = *(std::max_element(allJoints.begin(), allJoints.end()));

		mBindPoseTranslation.resize(mMaxJointID + 1);
		mBindPoseScale.resize(mMaxJointID + 1);
		mBindPoseRotator.resize(mMaxJointID + 1);

		mBindPosePreRotator.resize(mMaxJointID + 1);

		for (auto it : mAllJoints)
		{
			mBindPoseTranslation[it] = bindPoseTranslation[it];
			mBindPoseScale[it] = bindPoseScale[it];
			mBindPoseRotator[it] = bindPoseRotation[it];

			mBindPosePreRotator[it] = bindPosePreRotation[it];
		}
	}

	std::vector<Quat1f> JointAnimationInfo::getJointsRotation(float time)
	{
		updateJointsTransform(time);
		return mQuatRotation;
	}

	std::vector<Vec3f>  JointAnimationInfo::getJointsScale(float time)
	{
		updateJointsTransform(time);
		return mScale;
	}

	int JointAnimationInfo::findMaxSmallerIndex(const std::vector<float>& arr, float v) {
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



	Quat<Real> JointAnimationInfo::nlerp(const Quat<Real>& q1, const Quat<Real>& q2, float weight)
	{
		Quat1f tempQ;

		if (q1.x * q2.x < 0 && q1.y * q2.y < 0 && q1.z * q2.z < 0 && q1.w * q2.w < 0)
		{
			tempQ.x = -q2.x;
			tempQ.y = -q2.y;
			tempQ.z = -q2.z;
			tempQ.w = -q2.w;
		}
		else
		{
			tempQ = q2;
		}

		Quat<Real> result = (1 - weight) * q1 + weight * tempQ;
		// 归一化结果
		if (result.norm() < 0.001)
			result = Quat1f();
		else
			result.normalize();

		return result;
	}

	Quat<Real> JointAnimationInfo::slerp(const Quat<Real>& q1, const Quat<Real>& q2, float t)
	{
	
		// 计算内积
		double cosTheta = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;
		Quat<Real> result;

		if (abs(cosTheta) >= 1.0)
		{
			result.w = q1.w;
			result.x = q1.x;
			result.y = q1.y;
			result.z = q1.z;

			return result;
		}

		// 如果内积为负，取反q2，确保选择最短路径
		Quat<Real> q2Adjusted = q2;
		if (cosTheta < 0) {
			q2Adjusted.w = -q2.w;
			q2Adjusted.x = -q2.x;
			q2Adjusted.y = -q2.y;
			q2Adjusted.z = -q2.z;
			cosTheta = -cosTheta;
		}

		// 插值
		double theta = std::acos(cosTheta);
		double sinTheta = std::sin(theta);
		double weight1 = std::sin((1 - t) * theta) / sinTheta;
		double weight2 = std::sin(t * theta) / sinTheta;


		result.w = q1.w * weight1 + q2Adjusted.w * weight2;
		result.x = q1.x * weight1 + q2Adjusted.x * weight2;
		result.y = q1.y * weight1 + q2Adjusted.y * weight2;
		result.z = q1.z * weight1 + q2Adjusted.z * weight2;

		// 归一化结果
		if (result.norm() < 0.001)
			result = Quat1f();
		else
			result.normalize();

		return result;

	}

	std::vector<int> JointAnimationInfo::getJointDir(int Index, std::map<int, std::vector<int>> joint_Dir)
	{
		std::map<int, std::vector<int>>::const_iterator iter = joint_Dir.find(Index);
		if (iter == joint_Dir.end())
		{
			std::cout << "Error: not found JointIndex \n";

			std::vector<int> empty;
			return empty;
		}
		return iter->second;
	}

	Pose JointAnimationInfo::getPose(float inTime)
	{
		{
			float time = inTime;

			if (this->mLoop)
			{
				time = fmod(time, mTotalTime);
				//printf("Loop Clamp : %f\n", time);
			}
			//printf("total: %f\n",mTotalTime);
			//printf("time: %f\n",inTime);
			//printf("clampTime: %f\n", time);
			auto t = this->getJointsTranslation(time);
			auto s = this->getJointsScale(time);
			auto r = this->getJointsRotation(time);

			return Pose(t, s, r);
		}
	}

	Quat<Real> JointAnimationInfo::normalize(const Quat<Real>& q)
	{
		Real norm = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
		return { q.w / norm, q.x / norm, q.y / norm, q.z / norm };
	}


	Vec3f JointAnimationInfo::lerp(Vec3f v0, Vec3f v1, float weight)
	{
		return v0 + (v1 - v0) * weight;
	}

	Real JointAnimationInfo::lerp(Real v0, Real v1, float weight)
	{
		return v0 + (v1 - v0) * weight;
	}

	JointAnimationInfo::~JointAnimationInfo()
	{
		this->clear();

	};


}