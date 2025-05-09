
#pragma once
#include <vector>
#include <memory>
#include <string>
#include "Vector.h"
#include "OBase.h"
#include "Module.h"


namespace dyno {

	std::vector<int> getJointDirByJointIndex(int Index, std::map<int, std::vector<int>> jointId_joint_Dir);


	struct Pose
	{
		Pose() {}

		Pose(std::vector<Vec3f> t, std::vector<Vec3f> s, std::vector<Quat1f> r)
		{
			mTranslation = t;
			mQuatRotation = r;
			mScale = s;
		}

		~Pose()
		{
			mTranslation.clear();
			mScale.clear();
			mQuatRotation.clear();
		}

		int size() { return mTranslation.size(); }

		void resize(int s)
		{
			mTranslation.resize(s);
			mQuatRotation.resize(s);
			mScale.resize(s);
		};

		Pose& operator=(const Pose& other) {
			if (this != &other) {
				mTranslation = other.mTranslation;
				mQuatRotation = other.mQuatRotation;
				mScale = other.mScale;
			}
			return *this;
		}

		std::vector<Vec3f> mTranslation;
		std::vector<Quat1f> mQuatRotation;
		std::vector<Vec3f> mScale;
	};

	class JointInfo : public OBase
	{
		typedef int joint;

	public:

		JointInfo() {};

		JointInfo(
			DArray<Mat4f>& InverseBindMatrix,
			DArray<Mat4f>& LocalMatrix,
			DArray<Mat4f>& WorldMatrix,
			std::vector<int>& allJoints,
			std::map<joint, std::vector<joint>>& jointDir,
			std::map<joint, Vec3f>& bindPoseTranslation,
			std::map<joint, Vec3f>& bindPoseScale,
			std::map<joint, Quat1f>& bindPoseRotation
		);

		~JointInfo();

		void setGltfJointInfo(
			DArray<Mat4f>& InverseBindMatrix,
			std::vector<Mat4f>& LocalMatrix,
			DArray<Mat4f>& WorldMatrix,
			std::vector<int>& allJoints,
			std::map<joint, std::vector<joint>>& jointDir,
			std::map<joint, Vec3f>& bindPoseTranslation,
			std::map<joint, Vec3f>& bindPoseScale,
			std::map<joint, Quat1f>& bindPoseRotation
		);
		
		void clear() 
		{
			mJointName.clear();


			mBindPoseTranslation.clear();
			mBindPoseScale.clear();
			mBindPoseRotation.clear();
			mBindPoseRotator.clear();

			mAllJoints.clear();
			mJointDir.clear();

			mMaxJointID = -1;
		}

		void SetJointInfo(
			std::vector<Mat4f>& InverseBindMatrix,
			std::vector<Mat4f>& LocalMatrix,
			std::vector<Mat4f>& WorldMatrix,
			std::vector<int>& allJoints,
			std::map<joint, std::vector<joint>>& jointDir,
			std::map<joint, Vec3f>& bindPoseTranslation,
			std::map<joint, Vec3f>& bindPoseScale,
			std::map<joint, Vec3f>& bindPoseRotation,
			std::map<joint, Vec3f>& bindPosePreRotation
		);


		void setJoint(const JointInfo& j);

		bool isEmpty();

		void updateWorldMatrixByTransform();


		void setJointName(const std::map<int,std::string> name) { this->mJointName = name; }

		void setLeftHandedCoordSystem(bool islLeft) { useLeftHandedCoordSystem = islLeft; };

		void setPose(Pose pose) 
		{
			this->currentPose = pose;
			updateWorldMatrixByTransform();
		};


		int findJointIndexByName(const std::string& value)
		{
			for (const auto& pair : mJointName)
			{
				if (pair.second == value)
				{
					return pair.first; // 找到返回 key
				}
			}
			return -1; // 未找到
		}

		std::vector<Mat4f> getLocalMatrix(Pose& pose)
		{
			std::vector<Mat4f> localMatrix(this->mMaxJointID + 1);

			auto translation = pose.mTranslation;
			auto rotation = pose.mQuatRotation;
			auto scale = pose.mScale;

			for (size_t i = 0; i < mAllJoints.size(); i++)
			{
				int joint = mAllJoints[i];

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

			return localMatrix;
		}

	public:

		std::map<int, std::string> mJointName;

		DArray<Mat4f> mJointInverseBindMatrix;
		std::vector<Mat4f> mJointLocalMatrix;
		DArray<Mat4f> mJointWorldMatrix;


		
		Pose currentPose;

		std::vector<Vec3f>  mBindPoseTranslation;
		std::vector<Vec3f>  mBindPoseScale;
		std::vector<Quat<Real>> mBindPoseRotation;
		std::vector<Vec3f> mBindPoseRotator;

		std::vector<Vec3f> mBindPosePreRotator;
		//动画：
		
		std::vector<joint> mAllJoints;
		std::map<joint, std::vector<joint>> mJointDir;

		int mMaxJointID = -1;
	
	private:
		bool useLeftHandedCoordSystem = true;
	};


	class JointAnimationInfo : public OBase
	{
		typedef int joint;

	public:

		JointAnimationInfo() {};

		~JointAnimationInfo();

		void setGLTFAnimationData(
			std::map<joint, std::vector<Vec3f>>& jointTranslation,
			std::map<joint, std::vector<Real>>& jointTimeCodeTranslation,
			std::map<joint, std::vector<Vec3f>>& jointScale,
			std::map<joint, std::vector<Real>>& jointIndexTimeCodeScale,
			std::map<joint, std::vector<Quat1f>>& jointRotation,
			std::map<joint, std::vector<Real>>& jointIndexRotation,
			std::shared_ptr<JointInfo> skeleton,
			bool loop = true
		);


		void clear() 
		{
			mSkeleton = NULL;

			mJoint_KeyId_QuatRotation.clear();
			mJoint_KeyId_tQuatRotation.clear();

			mJoint_KeyId_T_X.clear();
			mJoint_KeyId_tT_X.clear();
			mJoint_KeyId_T_Y.clear();
			mJoint_KeyId_tT_Y.clear();
			mJoint_KeyId_T_Z.clear();
			mJoint_KeyId_tT_Z.clear();

			mJoint_KeyId_R_X.clear();
			mJoint_KeyId_tR_X.clear();
			mJoint_KeyId_R_Y.clear();
			mJoint_KeyId_tR_Y.clear();
			mJoint_KeyId_R_Z.clear();
			mJoint_KeyId_tR_Z.clear();
			
			mJoint_KeyId_S_X.clear();
			mJoint_KeyId_tS_X.clear();
			mJoint_KeyId_S_Y.clear();
			mJoint_KeyId_tS_Y.clear();
			mJoint_KeyId_S_Z.clear();
			mJoint_KeyId_tS_Z.clear();

			mTranslation.clear();
			mScale.clear();
			mQuatRotation.clear();
			mRotator.clear();

			mJointWorldMatrix.clear();

			mTotalTime = 0;
			currentTime = -1;
			mLoop = true;
			mBlendInTime = 0.0f;
			mBlendOutTime = 0.0f;
			mPlayRate = 1.0f;

			mAnimationTime = 0.0f;
		}

		bool isValid() 
		{
			return mTotalTime > 0;
		}
		bool isGltfAnimation()
		{
			return mJoint_KeyId_QuatRotation.size();
		}

		void resizeJointsData(int size)
		{
			mTranslation.resize(size);
			mScale.resize(size);
			mQuatRotation.resize(size);
			mRotator.resize(size);
		}



		std::vector<Vec3f> getJointsTranslation(float time);

		std::vector<Quat1f> getJointsRotation(float time);

		std::vector<Vec3f> getJointsScale(float time);
	
		float getTotalTime() { return mTotalTime; }
		
		int findMaxSmallerIndex(const std::vector<float>& arr, float v);

		Vec3f lerp(Vec3f v0, Vec3f v1, float weight);

		Quat<Real> normalize(const Quat<Real>& q);

		Quat<Real> slerp(const Quat<Real>& q1, const Quat<Real>& q2, float weight);
		Real lerp(Real v0, Real v1, float weight);

		Quat<Real> nlerp(const Quat<Real>& q1, const Quat<Real>& q2, float weight);

		std::vector<int> getJointDir(int Index, std::map<int, std::vector<int>> joint_Dir);

		void setLoop(bool loop) { mLoop = loop; }

		Pose getPose(float inTime);

		void updateAnimationPose(float inTime){

			auto pose = this->getPose(inTime);
			mSkeleton->setPose(pose);
			
		};

		float getCurrentAnimationTime() { return currentTime; }

		float& getBlendInTime() { return mBlendInTime; }

		float& getBlendOutTime() { return mBlendOutTime; }

		float& getPlayRate() { return mPlayRate; }

		void updateTotalTime();

		std::shared_ptr<JointInfo>& getSkeleton() { return mSkeleton; }

		void setSkeleton(std::shared_ptr<JointInfo> sk) { mSkeleton = sk; }

	private:

		void updateJointsTransform(float time);
		Transform3f updateTransform(joint jId, float time);

		Real calculateMinTime(const std::map<joint, std::vector<Real>>& timeCodes);
		Real calculateMaxTime(const std::map<joint, std::vector<Real>>& timeCodes);

		void offsetTimeCodes(std::map<joint, std::vector<Real>>& timeCodes, Real offset) {
			for (auto& it : timeCodes) {
				for (auto& time : it.second) {
					time -= offset;
				}
			}
		}

		void updateGLTFRotation(int select, Real time) 
		{
			auto iterGLTFR = mJoint_KeyId_QuatRotation.find(select);
			if (iterGLTFR == mJoint_KeyId_QuatRotation.end())
				mQuatRotation[select] = mSkeleton->mBindPoseRotation[select];
			{
				//Rotation
				if (iterGLTFR != mJoint_KeyId_QuatRotation.end())
				{
					const std::vector<Quat1f>& all_R = mJoint_KeyId_QuatRotation[select];
					const std::vector<Real>& tTimeCode = mJoint_KeyId_tQuatRotation[select];

					int tId = findMaxSmallerIndex(tTimeCode, time);

					if (tId >= all_R.size() - 1)				//   [size-1]<=[tId]  
					{
						mQuatRotation[select] = all_R[all_R.size() - 1];
					}
					else
					{
						float weight = (time - tTimeCode[tId]) / (tTimeCode[tId + 1] - tTimeCode[tId]);
						mQuatRotation[select] = nlerp(all_R[tId], all_R[tId + 1], weight);
					}
				}
			}	
		}


		void updateCurrentPose(int select,Real time, std::vector<Vec3f>& currentData,std::vector<Vec3f>& bindPoseData,std::map<joint, std::vector<Real>>& animationData,std::map<joint, std::vector<Real>>& TimeCode,int channel)
		{

			auto iter = animationData.find(select);
			if (iter == animationData.end())
				currentData[select][channel] = bindPoseData[select][channel];

			if (iter != animationData.end())
			{
				const std::vector<Real>& all_R = animationData[select];
				const std::vector<Real>& tTimeCode = TimeCode[select];

				int tId = findMaxSmallerIndex(tTimeCode, time);

				if (tId >= all_R.size() - 1)				//   [size-1]<=[tId]  
				{
					currentData[select][channel] = all_R[all_R.size() - 1];
				}
				else
				{
					float weight = (time - tTimeCode[tId]) / (tTimeCode[tId + 1] - tTimeCode[tId]);
					currentData[select][channel] = lerp(all_R[tId], all_R[tId + 1], weight);
				}
			}

		}

	public:		
		
		std::shared_ptr<JointInfo> mSkeleton = NULL;

		//Only Gltf
		std::map<joint, std::vector<Quat1f>> mJoint_KeyId_QuatRotation;
		std::map<joint, std::vector<Real>> mJoint_KeyId_tQuatRotation;

		//fbx Translation
		std::map<joint, std::vector<Real>> mJoint_KeyId_T_X;
		std::map<joint, std::vector<Real>> mJoint_KeyId_tT_X;
		std::map<joint, std::vector<Real>> mJoint_KeyId_T_Y;
		std::map<joint, std::vector<Real>> mJoint_KeyId_tT_Y;
		std::map<joint, std::vector<Real>> mJoint_KeyId_T_Z;
		std::map<joint, std::vector<Real>> mJoint_KeyId_tT_Z;
		//fbx Rotation
		std::map<joint, std::vector<Real>> mJoint_KeyId_R_X;
		std::map<joint, std::vector<Real>> mJoint_KeyId_tR_X;
		std::map<joint, std::vector<Real>> mJoint_KeyId_R_Y;
		std::map<joint, std::vector<Real>> mJoint_KeyId_tR_Y;
		std::map<joint, std::vector<Real>> mJoint_KeyId_R_Z;
		std::map<joint, std::vector<Real>> mJoint_KeyId_tR_Z;
		//fbx Scale
		std::map<joint, std::vector<Real>> mJoint_KeyId_S_X;
		std::map<joint, std::vector<Real>> mJoint_KeyId_tS_X;
		std::map<joint, std::vector<Real>> mJoint_KeyId_S_Y;
		std::map<joint, std::vector<Real>> mJoint_KeyId_tS_Y;
		std::map<joint, std::vector<Real>> mJoint_KeyId_S_Z;
		std::map<joint, std::vector<Real>> mJoint_KeyId_tS_Z;

	private:
		//当前时间下的动画数据，在某些情况下仅记录三维软件中具有动画变化的骨骼
		std::vector<Vec3f> mTranslation;
		std::vector<Vec3f> mScale;
		std::vector<Quat1f> mQuatRotation;
		std::vector<Vec3f> mRotator;

		DArray<Mat4f> mJointWorldMatrix;

		Real mTotalTime = 0;

		Real currentTime = -1;

		bool mLoop = true;
		Real mBlendInTime = 0.0f;
		Real mBlendOutTime = 0.0f;
		Real mPlayRate = 1.0f;

		Real mAnimationTime = 0.0f;

	};


	


	
}

