
#pragma once
#include <vector>
#include <memory>
#include <string>
#include "Vector.h"
#include "OBase.h"
#include "Module.h"


namespace dyno {

	struct Pose
	{
		Pose() {}

		Pose(std::vector<Vec3f> t, std::vector<Vec3f> s, std::vector<Quat1f> r)
		{
			mTranslation = t;
			mRotation = r;
			mScale = s;
		}

		~Pose()
		{
			mTranslation.clear();
			mScale.clear();
			mRotation.clear();
		}

		int size() { return mTranslation.size(); }

		void resize(int s)
		{
			mTranslation.resize(s);
			mRotation.resize(s);
			mScale.resize(s);
		};

		std::vector<Vec3f> mTranslation;
		std::vector<Quat1f> mRotation;
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

		void UpdateJointInfo(
			DArray<Mat4f>& InverseBindMatrix,
			DArray<Mat4f>& LocalMatrix,
			DArray<Mat4f>& WorldMatrix,
			std::vector<int>& allJoints,
			std::map<joint, std::vector<joint>>& jointDir,
			std::map<joint, Vec3f>& bindPoseTranslation,
			std::map<joint, Vec3f>& bindPoseScale,
			std::map<joint, Quat1f>& bindPoseRotation
		);

		void setJoint(const JointInfo& j);

		bool isEmpty();

		void updateWorldMatrixByTransform();

		void updateCurrentPose(std::map<joint, Vec3f> t, std::map<joint, Vec3f> s, std::map<joint, Quat1f> r);

		void setJointName(const std::map<int,std::string> name) { this->mJointName = name; }

	public:

		std::map<int, std::string> mJointName;

		DArray<Mat4f> mJointInverseBindMatrix;
		DArray<Mat4f> mJointLocalMatrix;
		DArray<Mat4f> mJointWorldMatrix;

		std::vector<Vec3f>  mBindPoseTranslation;
		std::vector<Vec3f>  mBindPoseScale;
		std::vector<Quat<Real>> mBindPoseRotation;

		//动画：
		
		DArray<Vec3f> mCurrentTranslation;
		DArray<Quat<Real>> mCurrentRotation;
		DArray<Vec3f> mCurrentScale;

		std::vector<joint> mAllJoints;
		std::map<joint, std::vector<joint>> mJointDir;

		int mMaxJointID = -1;
	};


	class JointAnimationInfo : public OBase
	{
		//这个结构要在开始时候直接把所有动画数据读入内存，供给动画状态机使用。
		typedef int joint;

	public:

		JointAnimationInfo() {};

		~JointAnimationInfo();

		void setAnimationData(
			std::map<joint, std::vector<Vec3f>>& jointTranslation,
			std::map<joint, std::vector<Real>>& jointTimeCodeTranslation,
			std::map<joint, std::vector<Vec3f>>& jointScale,
			std::map<joint, std::vector<Real>>& jointIndexTimeCodeScale,
			std::map<joint, std::vector<Quat1f>>& jointRotation,
			std::map<joint, std::vector<Real>>& jointIndexRotation,
			std::shared_ptr<JointInfo> skeleton
		);


		void updateJointsTransform(float time);

		Transform3f updateTransform(joint jId, float time); 

		std::vector<Vec3f> getJointsTranslation(float time);

		std::vector<Quat1f> getJointsRotation(float time);

		std::vector<Vec3f> getJointsScale(float time);
	
		float getTotalTime() { return mTotalTime; }
		
		int findMaxSmallerIndex(const std::vector<float>& arr, float v);

		Vec3f lerp(Vec3f v0, Vec3f v1, float weight);

		Quat<Real> normalize(const Quat<Real>& q);

		Quat<Real> slerp(const Quat<Real>& q1, const Quat<Real>& q2, float weight);

		Quat<Real> nlerp(const Quat<Real>& q1, const Quat<Real>& q2, float weight);

		std::vector<int> getJointDir(int Index, std::map<int, std::vector<int>> joint_Dir);

		void setLoop(bool loop) { mLoop = loop; }

		Pose getPose(float inTime);

		float getCurrentAnimationTime() { return currentTime; }

		float& getBlendInTime() { return mBlendInTime; }

		float& getBlendOutTime() { return mBlendOutTime; }

		float& getPlayRate() { return mPlayRate; }

	public:		
		
		std::shared_ptr<JointInfo> mSkeleton = NULL;

	private:

		//动画及时间戳
		std::map<joint, std::vector<Vec3f>> mJoint_Index_Translation;
		std::map<joint, std::vector<Real>> mJoint_Index_TimeCode_Translation;

		std::map<joint, std::vector<Vec3f>> mJoint_Index_Scale;
		std::map<joint, std::vector<Real>> mJoint_Index_TimeCode_Scale;

		std::map<joint, std::vector<Quat1f>> mJoint_Index_Rotation;
		std::map<joint, std::vector<Real>> mJoint_Index_TimeCode_Rotation;

		//当前时间下的动画数据，在某些情况下仅记录三维软件中具有动画变化的骨骼
		std::vector<Vec3f> mTranslation;
		std::vector<Vec3f> mScale;
		std::vector<Quat1f> mRotation;

		DArray<Mat4f> mJointWorldMatrix;

		float mTotalTime = 0;

		float currentTime = -1;

		bool mLoop = true;
		float mBlendInTime = 0.0f;
		float mBlendOutTime = 0.0f;
		float mPlayRate = 1.0f;

		float mAnimationTime = 0.0f;

	};


	


	
}

