
#include "JointInfo.h"
#include "GltfFunc.h"
#include "Matrix.h"

namespace dyno
{

	template< typename Vec3f, typename Quat1f ,typename Mat4f>
	__global__ void updateLocalMatrix(//需要把原始绑定算进来
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

		//printf("********** joint : %d  ***********\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n ***********************\nQuat : %f,%f,%f,%f\nScale : %f,%f,%f\nTrans : %f,%f,%f\n***********************\n",
		//	joint,
		//	c(0,0), c(0, 1), c(0, 2), c(0, 3),
		//	c(1, 0), c(1, 1), c(1, 2), c(1, 3),
		//	c(2, 0), c(2, 1), c(2, 2), c(2, 3),
		//	c(3, 0), c(3, 1), c(3, 2), c(3, 3),

		//	rotation[joint].x, rotation[joint].y, rotation[joint].z, rotation[joint].w,
		//	scale[joint][0], scale[joint][1], scale[joint][2],
		//	translation[joint][0], translation[joint][1], translation[joint][2]
		//	);

		

	}

	void JointInfo::updateWorldMatrixByTransform()
	{
		DArray<int> jointIds;
		jointIds.assign(mAllJoints);
		CArray<Vec3f> tt;
		CArray<Vec3f> ss;
		CArray<Quat1f> rr;
		CArray<Mat4f> mm;
		tt.assign(mCurrentTranslation);
		ss.assign(mCurrentScale);
		rr.assign(mCurrentRotation);
		mm.assign(mJointLocalMatrix);

		mCurrentTranslation.assign(tt);
		mCurrentScale.assign(ss);
		mCurrentRotation.assign(rr);
		mJointLocalMatrix.assign(mm);
		mJointLocalMatrix.resize(mCurrentTranslation.size());

		cuExecute(mAllJoints.size(),
			updateLocalMatrix,
			mCurrentTranslation,
			mCurrentScale,
			mCurrentRotation,
			mJointLocalMatrix,
			jointIds
		);


		std::vector<Mat4f> c_joint_Mat4f;
		c_joint_Mat4f.resize(mMaxJointID + 1);

		CArray<Mat4f> c_JointLocalMatrix;
		c_JointLocalMatrix.assign(mJointLocalMatrix);

		for (size_t i = 0; i < mAllJoints.size(); i++)
		{
			joint jointId = mAllJoints[i];
			const std::vector<int>& jD = getJointDirByJointIndex(jointId, mJointDir);


			Mat4f tempMatrix = Mat4f::identityMatrix();
			//
			for (int k = jD.size() - 1; k >= 0; k--)
			{
				joint select = jD[k];
				tempMatrix *= c_JointLocalMatrix[select];		//
			}
			c_joint_Mat4f[jointId] = tempMatrix;

		}

		mJointWorldMatrix.assign(c_joint_Mat4f);

		CArray<Mat4f> c_JointMatrix;
		c_JointMatrix.assign(mJointWorldMatrix);

		//for (size_t i = 0; i < c_JointMatrix.size(); i++)
		//{
		//	auto c = c_JointMatrix[i];

		//	printf("%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n",
		//		c(0,0), c(0, 1), c(0, 2), c(0, 3),
		//		c(1, 0), c(1, 1), c(1, 2), c(1, 3),
		//		c(2, 0), c(2, 1), c(2, 2), c(2, 3),
		//		c(3, 0), c(3, 1), c(3, 2), c(3, 3)
		//		);

		//}
	}


	void JointInfo::updateCurrentPose(std::map<joint, Vec3f> t, std::map<joint, Vec3f> s, std::map<joint, Quat1f> r)
	{





	}

	void JointInfo::setJoint(const JointInfo& j)
	{
		mJointInverseBindMatrix.assign(j.mJointInverseBindMatrix);
		mJointLocalMatrix.assign(j.mJointLocalMatrix);
		mJointWorldMatrix.assign(j.mJointWorldMatrix);
		mAllJoints = j.mAllJoints;
		mJointDir = j.mJointDir;
		mMaxJointID = j.mMaxJointID;
		mBindPoseTranslation = j.mBindPoseTranslation;
		mBindPoseScale = j.mBindPoseScale;
		mBindPoseRotation = j.mBindPoseRotation;
	}

	bool JointInfo::isEmpty()
	{
		if (mJointInverseBindMatrix.isEmpty() || mJointLocalMatrix.isEmpty() || mJointWorldMatrix.isEmpty())
			return true;
	}

	void JointInfo::UpdateJointInfo(
		DArray<Mat4f>& InverseBindMatrix,
		DArray<Mat4f>& LocalMatrix,
		DArray<Mat4f>& WorldMatrix,
		std::vector<int>& allJoints,
		std::map<joint, std::vector<joint>>& jointDir,
		std::map<joint, Vec3f>& bindPoseTranslation,
		std::map<joint, Vec3f>& bindPoseScale,
		std::map<joint, Quat1f>& bindPoseRotation
	)
	{
		mJointInverseBindMatrix.assign(InverseBindMatrix);
		mJointLocalMatrix.assign(LocalMatrix);
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
		mJointInverseBindMatrix.clear();
		mJointInverseBindMatrix.clear();
		mJointLocalMatrix.clear();
		mJointWorldMatrix.clear();

		mBindPoseTranslation.clear();
		mBindPoseScale.clear();
		mBindPoseRotation.clear();

		mCurrentTranslation.clear();
		mCurrentRotation.clear();
		mCurrentScale.clear();

		mAllJoints.clear();
		mJointDir.clear();

	};

	JointInfo::JointInfo(
		DArray<Mat4f>& InverseBindMatrix,
		DArray<Mat4f>& LocalMatrix,
		DArray<Mat4f>& WorldMatrix,
		std::vector<int>& allJoints,
		std::map<joint, std::vector<joint>>& jointDir,
		std::map<joint, Vec3f>& bindPoseTranslation,
		std::map<joint, Vec3f>& bindPoseScale,
		std::map<joint, Quat1f>& bindPoseRotation
	) 
	{
		UpdateJointInfo(InverseBindMatrix,
			LocalMatrix,
			WorldMatrix,
			allJoints,
			jointDir,
			bindPoseTranslation,
			bindPoseScale,
			bindPoseRotation
		);
	}


	void JointAnimationInfo::setAnimationData(
		std::map<joint, std::vector<Vec3f>>& jointTranslation,
		std::map<joint, std::vector<Real>>& jointTimeCodeTranslation,
		std::map<joint, std::vector<Vec3f>>& jointScale,
		std::map<joint, std::vector<Real>>& jointIndexTimeCodeScale,
		std::map<joint, std::vector<Quat1f>>& jointRotation,
		std::map<joint, std::vector<Real>>& jointIndexRotation,
		std::shared_ptr<JointInfo> skeleton
	)
	{

		mJoint_Index_Translation = jointTranslation;
		mJoint_Index_TimeCode_Translation = jointTimeCodeTranslation;

		mJoint_Index_Scale = jointScale;
		mJoint_Index_TimeCode_Scale = jointIndexTimeCodeScale;

		mJoint_Index_Rotation = jointRotation;
		mJoint_Index_TimeCode_Rotation = jointIndexRotation;
		mSkeleton = skeleton;

		mTranslation.resize(mSkeleton->mMaxJointID + 1);
		mScale.resize(mSkeleton->mMaxJointID + 1);
		mRotation.resize(mSkeleton->mMaxJointID + 1);

	

		float startR = NULL;
		float endR = NULL;
		for (auto it : mJoint_Index_TimeCode_Rotation )
		{ 
			{
				float tempMin = *std::min_element(it.second.begin(), it.second.end());

				if (startR == NULL)
					startR = tempMin;
				else
					startR = startR < tempMin ? startR : tempMin;
			}

			{
				float tempMax = *std::max_element(it.second.begin(), it.second.end());

				if (endR == NULL)
					endR = tempMax;
				else
					endR = endR > tempMax ? endR : tempMax;
			}
		}

		float startT = NULL;
		float endT = NULL;
		for (auto it : mJoint_Index_TimeCode_Translation)
		{
			{
				float tempMin = *std::min_element(it.second.begin(), it.second.end());

				if (startT == NULL)
					startT = tempMin;
				else
					startT = startT < tempMin ? startT : tempMin;
			}

			{
				float tempMax = *std::max_element(it.second.begin(), it.second.end());

				if (endT == NULL)
					endT = tempMax;
				else
					endT = endT > tempMax ? endT : tempMax;
			}
		}

		float startS = NULL;
		float endS = NULL;
		for (auto it : mJoint_Index_TimeCode_Scale)
		{
			{
				float tempMin = *std::min_element(it.second.begin(), it.second.end());

				if (startS == NULL)
					startS = tempMin;
				else
					startS = startS < tempMin ? startS : tempMin;
			}

			{
				float tempMax = *std::max_element(it.second.begin(), it.second.end());

				if (endS == NULL)
					endS = tempMax;
				else
					endS = endS > tempMax ? endS : tempMax;
			}
		}

		float timeMin = (startT < startR ? startT : startR) < startS ? (startT < startR ? startT : startR) : startS;
		float timeMax = (endT > endR ? endT : endR) > endS ? (endT > endR ? endT : endR) : endS;

		this->mTotalTime = timeMax - timeMin;
		printf("mTotalTime : %f\n", mTotalTime);
		// Offset TimeCode to "0 - mTotalTime"
		if (timeMin != 0) 
		{
			for (auto it : mJoint_Index_TimeCode_Rotation)
				for (size_t i = 0; i < it.second.size(); i++)
				{
					it.second[i] = it.second[i] - timeMin;
				}
			for (auto it : mJoint_Index_TimeCode_Translation)
				for (size_t i = 0; i < it.second.size(); i++)
				{
					it.second[i] = it.second[i] - timeMin;
				}
			for (auto it : mJoint_Index_TimeCode_Scale)
				for (size_t i = 0; i < it.second.size(); i++)
				{
					it.second[i] = it.second[i] - timeMin;
				}
		}


		printf("*******************\nAnimation Time : \nT:%f - %f \nS:%f - %f \nR:%f - %f \n*******************\n", startT, endT, startS, endS, startR, endR);

	};

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
		//基于循环添加time的裁切


		auto iterR = mJoint_Index_Rotation.find(select);
		if (iterR != mJoint_Index_Rotation.end())
			mRotation[select] = iterR->second[(int)time];
		else
			mRotation[select] = mSkeleton->mBindPoseRotation[select];

		{
			//Rotation
			if (iterR != mJoint_Index_Rotation.end())
			{
				const std::vector<Quat1f>& all_R = mJoint_Index_Rotation[select];
				const std::vector<Real>& tTimeCode = mJoint_Index_TimeCode_Rotation[select];

				int tId = findMaxSmallerIndex(tTimeCode, time);

				if (tId >= all_R.size() - 1)				//   [size-1]<=[tId]  
				{
					mRotation[select] = all_R[all_R.size() - 1];
				}
				else
				{
					if (all_R[tId] != all_R[tId + 1])
					{
						float weight = (time - tTimeCode[tId]) / (tTimeCode[tId + 1] - tTimeCode[tId]);
						mRotation[select] = slerp(all_R[tId], all_R[tId + 1], weight);
					}
				}
			}
			else
			{
				mRotation[select] = mSkeleton->mBindPoseRotation[select];
			}
		}


		//Translation
		auto iterT = mJoint_Index_Translation.find(select);
		if (iterT != mJoint_Index_Translation.end())
			mTranslation[select] = iterT->second[(int)time];
		else
			mTranslation[select] = mSkeleton->mBindPoseTranslation[select];

		{
			//Translation
			if (iterT != mJoint_Index_Translation.end())
			{
				const std::vector<Vec3f>& all_T = mJoint_Index_Translation[select];
				const std::vector<Real>& tTimeCode = mJoint_Index_TimeCode_Translation[select];

				int tId = findMaxSmallerIndex(tTimeCode, time);


				if (tId >= all_T.size() - 1)				//   [size-1]<=[tId]   大于最后一个
				{
					mTranslation[select] = all_T[all_T.size() - 1];
					//printf("joint : %d  , %d 最左  T: %f, %f, %f\n", select, all_T.size() - 1, all_T[all_T.size() - 1]);
				}
				else
				{
					if (all_T[tId] != all_T[tId + 1])
					{
						float weight = (time - tTimeCode[tId]) / (tTimeCode[tId + 1] - tTimeCode[tId]);
						mTranslation[select] = lerp(all_T[tId], all_T[tId + 1], weight);


					}
					//printf("joint : %d  , %d - %d  T: %f, %f, %f\n", select, tId, tId + 1, mTranslation[select]);


				}
			}
			else
			{
				mTranslation[select] = mSkeleton->mBindPoseTranslation[select];
			}
		}

		//Scale
		auto iterS = mJoint_Index_Scale.find(select);
		if (iterS != mJoint_Index_Scale.end())
			mScale[select] = iterS->second[(int)time];
		else
			mScale[select] = mSkeleton->mBindPoseScale[select];

		{
			//Scale
			if (iterS != mJoint_Index_Scale.end())
			{
				const std::vector<Vec3f>& all_S = mJoint_Index_Scale[select];
				const std::vector<Real>& tTimeCode = mJoint_Index_TimeCode_Scale[select];

				int tId = findMaxSmallerIndex(tTimeCode, time);


				if (tId >= all_S.size() - 1)				//   [size-1]<=[tId]   大于最后一个
				{
					mScale[select] = all_S[all_S.size() - 1];
					//printf("joint : %d  , %d 最左  T: %f, %f, %f\n", select, all_T.size() - 1, all_T[all_T.size() - 1]);
				}
				else
				{
					if (all_S[tId] != all_S[tId + 1])
					{
						float weight = (time - tTimeCode[tId]) / (tTimeCode[tId + 1] - tTimeCode[tId]);
						mScale[select] = lerp(all_S[tId], all_S[tId + 1], weight);
					}
					//printf("joint : %d  , %d - %d  T: %f, %f, %f\n", select, tId, tId + 1, mTranslation[select]);
				}
			}
			else
			{
				mScale[select] = mSkeleton->mBindPoseScale[select];
			}
		}

		return Transform3f(mTranslation[select], mRotation[select].toMatrix3x3(), mScale[select]);

	};

	std::vector<Vec3f> JointAnimationInfo::getJointsTranslation(float time)
	{
		updateJointsTransform(time);
		return mTranslation;
	}

	std::vector<Quat1f> JointAnimationInfo::getJointsRotation(float time)
	{
		updateJointsTransform(time);
		return mRotation;
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
			printf("total: %f\n",mTotalTime);
			printf("time: %f\n",inTime);
			printf("clampTime: %f\n", time);
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

	JointAnimationInfo::~JointAnimationInfo()
	{
		mJoint_Index_Translation.clear();
		mJoint_Index_TimeCode_Translation.clear();
		mJoint_Index_Scale.clear();
		mJoint_Index_TimeCode_Scale.clear();
		mJoint_Index_Rotation.clear();
		mJoint_Index_TimeCode_Rotation.clear();
		mTranslation.clear();
		mScale.clear();
		mRotation.clear();
		mJointWorldMatrix.clear();
		mSkeleton = NULL;

	};


}