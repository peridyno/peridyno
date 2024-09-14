/**
 * Copyright 2022 Yuzhong Guo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Topology/TextureMesh.h"

#include "Module/ComputeModule.h"

#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"

#include "FilePath.h"
#include "SkinInfo.h"
#include "JointInfo.h"
#include "Topology/DiscreteElements.h"

#include "../Dynamics/Cuda/RigidBody/RigidBodySystem.h"
#include "Topology/PointSet.h"

namespace dyno
{


	/**
	 * @brief A class to facilitate showing the shape information
	 */
	class AnimationMachine;

	template<typename TDataType>
	class AnimationMixer : public Node
	{
		DECLARE_TCLASS(AnimationMixer, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename TopologyModule::Triangle Triangle;


		typedef unsigned char byte;
		typedef int joint;
		typedef int shape;
		typedef int mesh;
		typedef int primitive;
		typedef int scene;

		AnimationMixer();
		~AnimationMixer();

	public:

		DEF_VAR(int, JointNum, 0, "Weight");

		DEF_VAR(Vec3f, InitialForwardVector, Vec3f(0, 0, 1), "Forward");
		DEF_VAR(Vec3f, InitialRightVector, Vec3f(-1, 0, 0), "Right");

		DEF_VAR(bool, RotationToVelocity, true, "RotationToVelocity");
		DEF_VAR(Real, MaxSpeed, 3, "Speed");
		DEF_VAR(Real, MaxAcceleratedSpeed, 20, "AcceleratedSpeed");
		DEF_VAR(Real, BrakingSpeed, 10, "BrakingSpeed");

		DEF_INSTANCE_IN(JointAnimationInfo, Idle, "Idle");
		DEF_INSTANCE_IN(JointAnimationInfo, Walk, "Walk");

		DEF_VAR_STATE(Vec3f, AcceleratedSpeed, Vec3f(0, 0, 0), "Position");
		DEF_VAR_STATE(Vec3f, ForwardVector, Vec3f(0, 0, 1), "Position");
		DEF_VAR_STATE(Vec3f, RightVector, Vec3f(-1, 0, 0), "Position");
		DEF_VAR_STATE(Vec3f, Position, Vec3f(0), "Position");
		DEF_VAR_STATE(Mat3f, Rotation, Mat3f::identityMatrix(), "Rotation");
		DEF_VAR_STATE(Quat<Real>, Quat, Quat<Real>(), "Quat");
		DEF_VAR_STATE(Vec3f, Velocity, Vec3f(0), "Position");

		DEF_INSTANCE_IN(JointInfo, Skeleton, "Skeleton");

		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");
		DEF_INSTANCE_STATE(JointInfo, Joint, "Joint");

		DEF_ARRAYLIST_STATE(Transform3f, InstanceTransform, DeviceType::GPU, "Instance transforms");


		DEF_INSTANCE_STATE(DiscreteElements<TDataType>, Topology, "Topology");
		DEF_ARRAY_STATE(Real, ElementsMass, DeviceType::GPU, "Mass of rigid bodies");
		DEF_ARRAY_STATE(Vec3f, ElementsCenter, DeviceType::GPU, "Center of rigid bodies");
		DEF_ARRAY_STATE(Real, ElementsLength, DeviceType::GPU, "Center of rigid bodies");
		DEF_ARRAY_STATE(Real, ElementsRadius, DeviceType::GPU, "Center of rigid bodies");
		DEF_ARRAY_STATE(Vec3f, ElementsOffset, DeviceType::GPU, "Offset of barycenters");
		DEF_ARRAY_STATE(Vec3f, ElementsVelocity, DeviceType::GPU, "Velocity of rigid bodies");
		DEF_ARRAY_STATE(Vec3f, ElementsAngularVelocity, DeviceType::GPU, "Angular velocity of rigid bodies");
		DEF_ARRAY_STATE(Mat3f, ElementsRotationMatrix, DeviceType::GPU, "Rotation matrix of rigid bodies");
		DEF_ARRAY_STATE(Quat<Real>, ElementsQuaternion, DeviceType::GPU, "Quaternion");

		DEF_INSTANCE_STATE(PointSet<TDataType>, CenterTopology, "Topology");


	public:



	protected:
		void updateElements();
		void initialElements();

		void resetStates() override;

		void updateStates() override;

		void updateAnimation(const float& time, const float& weight);

		Pose mixPose(Pose a, Pose b, float weight);

		void setAxisValue(Vec3f value)
		{
			mInputAxisValue = value;
		}

		void move(Vec3f axisValue);

		Quat1f QuatNormalize(Quat<Real>& q);

		Quat1f slerp(const Quat<Real>& q1, const Quat<Real>& q2, float t);

		void updateInstanceTransform();


		void addPositionOffset(Vec3f dir, bool updateVelocity = true);

		Quat1f getQuat(Vec3f v0, Vec3f v1)
		{
			Quat1f r = Quat1f(v0, v1);
			if (v1 == Vec3f(0, 0, -1))
				r = Quat1f(M_PI, Vec3f(0, 1, 0));

			return r;
		}

		Vec3f projectVectorOntoPlane(const Vec3f& v, const Vec3f& n);

	private:

		//AnimationMachine mAnimationMachine; 

		std::vector<JointAnimationInfo> mAnimations;

		int mMaxJointID = -1;

		Vec3f mInputAxisValue = Vec3f(0);

		DArray<int> mJointStart;
		DArray<int> mJointEnd;
		DArray<Vec3f> mJointCenter;
		DArray<Quat<Real>> mJointQuat;
		DArray<Mat3f> mJointRotation;


	};


	//class AnimationMachine
	//{
	//	/*
	//	状态机：
	//		处理状态的激活标记、混入混出的权重、动画的进程、采样动画变换
	//	*/
	//public:
	//	AnimationMachine()
	//	{
	//	};

	//	~AnimationMachine() {};

	//	void addState(std::shared_ptr<JointAnimationInfo> state)
	//	{
	//		
	//	}
	//	void addConnection(std::shared_ptr<JointAnimationInfo> first, std::shared_ptr<JointAnimationInfo> second)
	//	{
	//		
	//	}

	//private:

	//	std::vector<std::shared_ptr<JointAnimationInfo>> mActiveStates;

	//	std::vector<std::shared_ptr<JointAnimationInfo>>  mAllStates;

	//	std::vector<std::pair<std::shared_ptr<JointAnimationInfo>, std::shared_ptr<JointAnimationInfo>>> mConnection;

	//};

	//class AnimationState 
	//{
	//	/*
	//	状态机：
	//		处理状态的激活标记、混入混出的权重、动画的进程、采样动画变换
	//	*/
	//public:
	//	AnimationState(std::shared_ptr<JointAnimationInfo> animation)
	//	{
	//		this->setAnimation(animation);
	//	};
	//	
	//	~AnimationState() {};

	//	void setBlendTime(float t) 
	//	{
	//		mBlendTime = t < mTotalTime ? t : mTotalTime;
	//	}

	//	void startState(float time) 
	//	{
	//		mActive = true;
	//		mActiveTime = time;
	//		mStateTime = 0;
	//		mMixing = true;
	//	}
	//	void updateAnimation(float time)
	//	{
	//		float blendWeight = (mBlendTime - mStateTime) > 0 ? mBlendTime : mStateTime;

	//	}
	//	void endState()
	//	{
	//		mActive = false;
	//		mActiveTime = -1;
	//		mStateTime = 0;
	//		mMixing = false;
	//	}
	//	void setAnimation(std::shared_ptr<JointAnimationInfo> animation,bool l = true) 
	//	{
	//		mTotalTime = animation->getTotalTime();
	//		mAnimation = animation;
	//		mLoop = l;
	//	}
	//	void setLoop(bool l) 
	//	{
	//		mLoop = l;
	//	}

	//	Pose getStatePose(float time) 
	//	{
	//		return getPose(mAnimation, fmod(time,mTotalTime));
	//	}

	//private:
	//	
	//	std::shared_ptr<JointAnimationInfo> mAnimation;

	//	float mTotalTime = 0;
	//	float mActiveTime = -1;
	//	float mStateTime = 0;	//currentAnimationTime
	//	
	//	bool mActive = false;
	//	bool mMixing = false;

	//	float mBlendTime = 0;
	//	bool mLoop = true;


	//};




	IMPLEMENT_TCLASS(AnimationMixer, TDataType);
}
