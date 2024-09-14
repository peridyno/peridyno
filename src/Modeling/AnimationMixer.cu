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

#include "AnimationMixer.h"
#include "CharacterController.h"
#include <GLPhotorealisticRender.h>
#include "GLPointVisualModule.h"


//#include "GltfFunc.h"

namespace dyno
{
	/**
	 * @brief A class to facilitate showing the shape information
	 */

	template<typename TDataType>
	AnimationMixer<TDataType>::AnimationMixer() : 
		Node()
	{

		this->stateJoint()->setDataPtr(std::make_shared<JointInfo>());


		this->stateJoint()->promoteOuput();

		auto controller = std::make_shared<CharacterController<DataType3f>>();
		this->stateTimeStep()->connect(controller->inDeltaTime());
		controller->setUpdateAlways(true);

		auto boundMemberFunction = std::bind(&AnimationMixer<TDataType>::setAxisValue,this, std::placeholders::_1);
		controller->mDispatcher.addEventListener("setAxisValue", boundMemberFunction);

		this->stateTopology()->setDataPtr(std::make_shared<DiscreteElements<DataType3f>>());

		this->animationPipeline()->pushModule(controller);
		this->stateTimeStep()->setValue(0.03);


		

		this->setForceUpdate(true);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		this->stateCenterTopology()->setDataPtr(std::make_shared<PointSet<DataType3f>>());
		ptRender->varPointSize()->setValue(0.01);
		ptRender->varBaseColor()->setValue(Color::Red());

		this->stateCenterTopology()->connect(ptRender->inPointSet());
		this->graphicsPipeline()->pushModule(ptRender);




	}

	template<typename TDataType>
	AnimationMixer<TDataType>::~AnimationMixer()
	{
		
	}

	template<typename TDataType>
	void AnimationMixer<TDataType>::resetStates()
	{
		auto outputJoint = this->stateJoint()->getDataPtr();
		auto sourceJoint = this->inSkeleton()->constDataPtr();

		if (sourceJoint != NULL)
		{
			outputJoint->setJoint(*sourceJoint);
		}


		this->updateInstanceTransform();

		this->stateAcceleratedSpeed()->setValue(Vec3f(0));
		this->stateForwardVector()->setValue(Vec3f(0,0,1));		
		this->stateRightVector()->setValue(Vec3f(-1, 0, 0));
		this->statePosition()->setValue(Vec3f(0, 0, 0));
		this->stateQuat()->setValue(Quat<Real>());
		this->stateRotation()->setValue(Mat3f::identityMatrix());
		this->stateVelocity()->setValue(Vec3f(0));


		this->initialElements();
	}

	template<typename TDataType>
	void AnimationMixer<TDataType>::updateStates()
	{
		Node::updateStates();
		//damping
		this->move(mInputAxisValue);

		float time = this->stateElapsedTime()->getData();
		float weight = this->stateVelocity()->getValue().norm()/this->varMaxSpeed()->getValue();

		this->updateAnimation(time,weight);


		auto controller = this->animationPipeline()->findFirstModule<CharacterController<DataType3f>>();

		this->updateElements();

		this->updateInstanceTransform();

	}

	template<typename TDataType>
	void AnimationMixer<TDataType>::updateAnimation(const float& time,const float& weight)
	{
		auto outputJoint = this->stateJoint()->getDataPtr();

		auto idle = this->inIdle()->constDataPtr();
		auto walk = this->inWalk()->constDataPtr();


		auto poseA = idle->getPose(time);
		auto poseB = walk->getPose(time);
		auto currentPose = mixPose(poseA, poseB, weight);
		//auto currentPose = mixPose(poseA, poseB, this->varWeight()->getValue());




		//output
		outputJoint->mCurrentTranslation.assign(currentPose.mTranslation);
		outputJoint->mCurrentScale.assign(currentPose.mScale);
		outputJoint->mCurrentRotation.assign(currentPose.mRotation);
		
		outputJoint->updateWorldMatrixByTransform();



	}

	template <typename Mat4f, typename Vec3f, typename Mat3f>
	__global__ void initialPrimitives(
		DArray<int> start,
		DArray<int> end,
		DArray<Mat4f> worldMatrix,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotation,
		DArray<Quat<Real>> quat,
		DArray<Real> mass,
		DArray<Vec3f> offset,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angularVelocity,
		DArray<Real> length,
		DArray<Real> radius,
		Vec3f characterPosition,
		Quat<Real> characterRotation
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= start.size()) return;

		int startId = start[pId];
		int endId = end[pId];

		Vec3f p1 = Vec3f(worldMatrix[startId](0, 3), worldMatrix[startId](1, 3), worldMatrix[startId](2, 3));
		Vec3f p2 = Vec3f(worldMatrix[endId](0, 3), worldMatrix[endId](1, 3), worldMatrix[endId](2, 3));

		Vec3f v1 = p1 - p2;
		length[pId] = v1.norm();
		radius[pId] = length[pId]/3;
		v1.normalize();

		pos[pId] = characterRotation.toMatrix3x3() * ((p1 + p2) / 2) + characterPosition;
		quat[pId] = characterRotation * Quat1f(Vec3f(0, 1, 0), v1);
		rotation[pId] = quat[pId].toMatrix3x3();

		mass[pId] = 100;
		offset[pId] = Vec3f(0);
		velocity[pId] = Vec3f(0);
		angularVelocity[pId] = Vec3f(0);

	}

	template <typename Mat4f, typename Vec3f, typename Mat3f>
	__global__ void updatePrimitives(
		DArray<int> start,
		DArray<int> end,
		DArray<Mat4f> worldMatrix,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotation,
		DArray<Quat<Real>> quat,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angularVelocity,
		Vec3f characterPosition,
		Quat<Real> characterRotation
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= start.size()) return;

		int startId = start[pId];
		int endId = end[pId];

		Vec3f lastPos = pos[pId];
		Quat1f q1 = quat[pId];


		Vec3f p1 = Vec3f(worldMatrix[startId](0, 3), worldMatrix[startId](1, 3), worldMatrix[startId](2, 3));
		Vec3f p2 = Vec3f(worldMatrix[endId](0, 3), worldMatrix[endId](1, 3), worldMatrix[endId](2, 3));

		Vec3f v1 = p1 - p2;
		v1.normalize();

		pos[pId] = characterRotation.toMatrix3x3()*((p1 + p2) / 2) + characterPosition;
		quat[pId] = characterRotation * Quat1f(Vec3f(0, 1, 0), v1);
		rotation[pId] = quat[pId].toMatrix3x3();

		
		velocity[pId] = pos[pId] - lastPos;
		Quat1f q2 = quat[pId];

		Quat1f deltaQ = Quat1f(q2.w * q1.w + q2.x * q1.x + q2.y * q1.y + q2.z * q1.z,
			q2.w * q1.x - q2.x * q1.w - q2.y * q1.z + q2.z * q1.y,
			q2.w * q1.y + q2.x * q1.z - q2.y * q1.w - q2.z * q1.x,
			q2.w * q1.z - q2.x * q1.y + q2.y * q1.x - q2.z * q1.w);

		Quat1f angularV = Quat1f(2 * deltaQ.x, 2 * deltaQ.y, 2 * deltaQ.z, 0.0);

		double angularVelocityNorm = angularV.norm();
		angularV.w = 0.0; 
		angularV.x /= angularVelocityNorm;
		angularV.y /= angularVelocityNorm;
		angularV.z /= angularVelocityNorm;

		angularVelocity[pId] = Vec3f(angularV.x,angularV.y,angularV.z);

	}

	template<typename TDataType>
	void AnimationMixer<TDataType>::updateElements() 
	{
		auto jointData = this->stateJoint()->constDataPtr();
		
		cuExecute(mJointStart.size(),
			updatePrimitives,
			mJointStart,
			mJointEnd,
			jointData->mJointWorldMatrix,
			this->stateElementsCenter()->getData(),
			this->stateElementsRotationMatrix()->getData(),
			this->stateElementsQuaternion()->getData(),
			this->stateElementsVelocity()->getData(),
			this->stateElementsAngularVelocity()->getData(),
			this->statePosition()->getData(),
			this->stateQuat()->getData()
		);
	}

	template <typename TDataType>
	Quat1f AnimationMixer<TDataType>::slerp(const Quat<Real>& q1, const Quat<Real>& q2, float t)
	{

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

		Quat<Real> q2Adjusted = q2;
		if (cosTheta < 0) {
			q2Adjusted.w = -q2.w;
			q2Adjusted.x = -q2.x;
			q2Adjusted.y = -q2.y;
			q2Adjusted.z = -q2.z;
			cosTheta = -cosTheta;
		}

		double theta = std::acos(cosTheta);
		double sinTheta = std::sin(theta);
		double weight1 = std::sin((1 - t) * theta) / sinTheta;
		double weight2 = std::sin(t * theta) / sinTheta;


		result.w = q1.w * weight1 + q2Adjusted.w * weight2;
		result.x = q1.x * weight1 + q2Adjusted.x * weight2;
		result.y = q1.y * weight1 + q2Adjusted.y * weight2;
		result.z = q1.z * weight1 + q2Adjusted.z * weight2;

		if (result.norm() < 0.001)
			result = Quat<Real>();
		else
			result.normalize();

		return result;
	}

	template<typename TDataType>
	void AnimationMixer<TDataType>::updateInstanceTransform()
	{
		auto texMesh = this->inTextureMesh()->getDataPtr();
		int shapeNum = texMesh->shapes().size();

		uint N = 0;
		if (!this->inTextureMesh()->isEmpty())
			N = texMesh->shapes().size();

		CArrayList<Transform3f> tms;
		CArray<uint> instanceNum(N);
		instanceNum.reset();

		//Calculate instance number
		for (uint i = 0; i < instanceNum.size(); i++)
		{
			instanceNum[i] = 1;
		}

		if (instanceNum.size() > 0)
			tms.resize(instanceNum);

		//Initialize CArrayList
		for (uint i = 0; i < N; i++)
		{
			for (uint j = 0; j < instanceNum[i]; j++)
			{
				tms[i].insert(Transform3f(this->statePosition()->getValue(), this->stateRotation()->getValue(), Vec3f(1)));
			}
		}

		this->stateInstanceTransform()->assign(tms);

	}
	

	template<typename TDataType>
	void AnimationMixer<TDataType>::initialElements() 
	{
		CArray<int> c_JointStart;
		CArray<int> c_JointEnd;
		

		auto jointData = this->stateJoint()->constDataPtr();
		for (auto it : jointData->mJointDir)
		{
			int start = it.first;
			printf("%d -",start);
			for (size_t i = 0; i < it.second.size(); i++)
			{
				printf("%d,",it.second[i]);
			}
			printf("\n");

			if (it.second.size() >= 2) 
			{
				int end = it.second[1];
				c_JointStart.pushBack(start);
				c_JointEnd.pushBack(end);
			}

		}

		mJointStart.assign(c_JointStart);
		mJointEnd.assign(c_JointEnd);


		this->stateElementsMass()->resize(c_JointStart.size());
		this->stateElementsCenter()->resize(c_JointStart.size());
		this->stateElementsOffset()->resize(c_JointStart.size());
		this->stateElementsVelocity()->resize(c_JointStart.size());
		this->stateElementsRotationMatrix()->resize(c_JointStart.size());
		this->stateElementsAngularVelocity()->resize(c_JointStart.size());
		this->stateElementsQuaternion()->resize(c_JointStart.size());
		this->stateElementsLength()->resize(c_JointStart.size());
		this->stateElementsRadius()->resize(c_JointStart.size());

		cuExecute(mJointStart.size(),
			initialPrimitives,
			mJointStart,
			mJointEnd,
			jointData->mJointWorldMatrix,
			this->stateElementsCenter()->getData(),
			this->stateElementsRotationMatrix()->getData(),
			this->stateElementsQuaternion()->getData(),
			this->stateElementsMass()->getData(),
			this->stateElementsOffset()->getData(),
			this->stateElementsVelocity()->getData(),
			this->stateElementsAngularVelocity()->getData(),
			this->stateElementsLength()->getData(),
			this->stateElementsRadius()->getData(),
			this->statePosition()->getData(),
			this->stateQuat()->getData()
		);

		this->stateCenterTopology()->getDataPtr()->setPoints(this->stateElementsCenter()->getData());


	}




	template<typename TDataType>
	Pose AnimationMixer<TDataType>::mixPose(Pose a, Pose b, float weight)
	{
		if (a.size() && b.size())
		{
			//lerp
			if (a.size() != b.size())
				return Pose();

			std::vector<Vec3f> outT;
			std::vector<Vec3f> outS;
			std::vector<Quat1f> outR;

			outT.resize(a.size());
			outS.resize(a.size());
			outR.resize(a.size());

			JointAnimationInfo* temp = new JointAnimationInfo;
			for (size_t i = 0; i < a.size(); i++)
			{
				outT[i] = temp->lerp(a.mTranslation[i], b.mTranslation[i], weight);
				outS[i] = temp->lerp(a.mScale[i], b.mScale[i], weight);
				outR[i] = temp->slerp(a.mRotation[i], b.mRotation[i], weight);
			}
			delete temp;

			return Pose(outT, outS, outR);
		}
		else
			return Pose();
	}

	template<typename TDataType>
	void AnimationMixer<TDataType>::addPositionOffset(Vec3f dir, bool updateVelocity = true)
	{
		auto dt = this->stateTimeStep()->getValue();
		auto p = this->statePosition()->getValue();

		Vec3f velocity = this->stateVelocity()->getValue();

		float maxSpeed = this->varMaxSpeed()->getValue();
		this->stateAcceleratedSpeed()->setValue(this->varMaxAcceleratedSpeed()->getValue() * dir);

		velocity = this->stateAcceleratedSpeed()->getValue() * dt + velocity;
		if (velocity.norm() > maxSpeed)
			velocity = velocity.normalize() * maxSpeed;


		//braking
		if (this->stateAcceleratedSpeed()->getValue().norm() < 0.0001)
		{
			float braking = this->varBrakingSpeed()->getValue();
			Vec3f tempVelocity = velocity;

			Vec3f brakingVec = -1 * Vec3f(velocity).normalize() * braking * dt;

			velocity = Vec3f(velocity.x + brakingVec.x, velocity.y + brakingVec.y, velocity.z + brakingVec.z);

			if (velocity.dot(tempVelocity) <= 0)
				velocity = Vec3f(0);
		}

		Vec3f target = p + velocity * dt;

		//Åö×²¼ì²âÓëÎ»ÖÃÐÞÕý




		this->statePosition()->setValue(target);

		if (updateVelocity)
			this->stateVelocity()->setValue((target - p) / dt);

	};

	template<typename TDataType>
	Vec3f AnimationMixer<TDataType>::projectVectorOntoPlane(const Vec3f& v, const Vec3f& n) 
	{

		double projScalar = v.dot(n) / n.normSquared();
		Vec3f proj = projScalar * n;

		Vec3f proj_vector = v - proj;
		return proj_vector;
	}


	template<typename TDataType>
	Quat1f AnimationMixer<TDataType>::QuatNormalize(Quat<Real>& q)
	{
		Real d = q.norm();
		if (d < 0.00001) {
			q.w = Real(1.0);
			q.x = q.y = q.z = Real(0.0);
			return q;
		}
		d = Real(1) / d;
		q.x *= d;
		q.y *= d;
		q.z *= d;
		q.w *= d;
		return q;
	}


	template<typename TDataType>
	void AnimationMixer<TDataType>::move(Vec3f axisValue)
	{
		JointAnimationInfo* temp = new JointAnimationInfo;
		//move
		addPositionOffset(axisValue.normalize(), true);

		//Rotation
		if (this->varRotationToVelocity()->getValue() && this->stateVelocity()->getValue().norm() > 0.001)
		{
			Vec3f v0 = Vec3f(0, 0, 1);
			Vec3f v1 = this->stateVelocity()->getValue().normalize();

			Vec3f projectV0 = projectVectorOntoPlane(v0, Vec3f(0, 1, 0)).normalize();
			Vec3f projectV1 = projectVectorOntoPlane(v1, Vec3f(0, 1, 0)).normalize();

			Quat1f targetRotation = getQuat(projectV0, projectV1);


			Quat1f qa = QuatNormalize(this->stateQuat()->getValue());
			Quat1f qb = QuatNormalize(targetRotation);

			Quat1f q = this->slerp(qa, qb, 0.2);

			printf("[%f,%f,%f,%f] - [%f,%f,%f,%f]\n", qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w);
			printf("V(%f,%f,%f) - V(%f,%f,%f)\n", projectV0.x, projectV0.y, projectV0.z, projectV1.x, projectV1.y, projectV1.z);
			printf("result : %f,%f,%f,%f\n", q.x, q.y, q.z, q.w);
			

			this->stateQuat()->setValue(q);

			this->stateRotation()->setValue(q.toMatrix3x3());

		}


		mInputAxisValue = Vec3f(0);
	}




	DEFINE_CLASS(AnimationMixer);
}