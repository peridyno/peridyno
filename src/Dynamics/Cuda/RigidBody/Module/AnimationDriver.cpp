#include "AnimationDriver.h"

namespace dyno
{
	IMPLEMENT_TCLASS(AnimationDriver, TDataType);

	template<typename TDataType>
	AnimationDriver<TDataType>::AnimationDriver()
		: KeyboardInputModule()
	{
		this->varCacheEvent()->setValue(false);
	}

	template<typename TDataType>
	void AnimationDriver<TDataType>::onEvent(PKeyboardEvent event)
	{
		auto AnimationInfo = this->inJointAnimationInfo()->constDataPtr();
		auto skeleton = AnimationInfo->getSkeleton();
		if (!skeleton) 
		{
			printf("Error : No Skeleton \n");
		}

		float inputTimeEnd = AnimationInfo->getTotalTime();

		auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->inTopology()->getDataPtr());

		auto& d_hinge = topo->hingeJoints();
		CArray<HingeJoint> c_hinge;
		c_hinge.assign(d_hinge);

		std::map<int, std::vector<std::vector<Real>*>> animRot;
		std::map<int, std::vector<std::vector<Real>*>> animTime;


		auto binding = this->varBindingConfiguration()->getValue();


		for (size_t i = 0; i < binding.size(); i++)
		{


			auto name = binding[i].JointName;
			auto hingeId = binding[i].JointId;

			auto boneId = skeleton->findJointIndexByName(name);
			if (boneId != -1)
			{
				std::vector<std::vector<Real>*> tempV(3);
				std::vector<std::vector<Real>*> tempT(3);
				animRot[hingeId] = tempV;
				animTime[hingeId] = tempT;

				animRot[hingeId][0] = &AnimationInfo->mJoint_KeyId_R_X[boneId];
				animTime[hingeId][0] = &AnimationInfo->mJoint_KeyId_tR_X[boneId];

				animRot[hingeId][1] = &AnimationInfo->mJoint_KeyId_R_Y[boneId];
				animTime[hingeId][1] = &AnimationInfo->mJoint_KeyId_tR_Y[boneId];

				animRot[hingeId][2] = &AnimationInfo->mJoint_KeyId_R_Z[boneId];
				animTime[hingeId][2] = &AnimationInfo->mJoint_KeyId_tR_Z[boneId];

			}
		}


		move += this->inDeltaTime()->getValue() * this->varSpeed()->getValue();

		float weight = 0;
		int keyFrame = -1;


		float angleDivede = -1.0;

		Real epsilonAngle = M_PI / 360;

		switch (event.key)
		{
		case PKeyboardType::PKEY_W:

			angleDivede = -1.0;
			break;

		case PKeyboardType::PKEY_S:

			angleDivede = 1.0;

			break;
		default:
			return;
			break;

		}

		auto lerp = [](float a, float b, float t) {
			return a + t * (b - a);
		};


		for (size_t i = 0; i < binding.size(); i++)
		{
			int hingeId = binding[i].JointId;
			int axis = binding[i].Axis;
			float intensity = binding[i].Intensity;

			if (!animRot[hingeId].size())
				continue;

			Vec3i keyFrame = Vec3i(-1);
			Vec3f weight = Vec3f(-1);

			getFrameAndWeight(move, keyFrame, weight,animTime[hingeId]);

			std::vector<Real>& animX = *animRot[hingeId][0];
			std::vector<Real>& animY = *animRot[hingeId][1];
			std::vector<Real>& animZ = *animRot[hingeId][2];

			std::vector<Real>& currentAnimation = animX;

			switch (axis)
			{
			case 0:
				currentAnimation = animX;
				break;
			case 1:
				currentAnimation = animY;
				break;
			case 2:
				currentAnimation = animZ;
				break;
			default:
				break;
			}

			Real angle = 0;
			if (weight[axis] == -1) {
				angle = currentAnimation[keyFrame[axis]] * M_PI / 180 / angleDivede;
			}
			else {
				angle = lerp(currentAnimation[keyFrame[axis]], currentAnimation[keyFrame[axis] + 1], weight[axis]) * M_PI / 180 / angleDivede * intensity;
			}

			c_hinge[hingeId].setRange(angle, angle + epsilonAngle);
		}

		d_hinge.assign(c_hinge);


	}


	
	DEFINE_CLASS(AnimationDriver);
}