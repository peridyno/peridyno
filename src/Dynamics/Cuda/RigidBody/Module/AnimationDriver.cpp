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
		auto hierarchicalScene = this->inHierarchicalScene()->constDataPtr();

		float inputTimeStart = hierarchicalScene->mTimeStart;
		float inputTimeEnd = hierarchicalScene->mTimeEnd;

		auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->inTopology()->getDataPtr());

		auto& d_hinge = topo->hingeJoints();
		CArray<HingeJoint> c_hinge;
		c_hinge.assign(d_hinge);

		std::vector<std::vector<Real>*> animRot; 
		std::vector<std::vector<Real>*> animTime;

		animRot.resize(c_hinge.size());
		animTime.resize(c_hinge.size());


		auto hinge_DriveObjName = this->varDriverName()->getValue();


		for (size_t i = 0; i < hinge_DriveObjName.size(); i++)
		{
			auto name = hinge_DriveObjName[i];
			auto bone = hierarchicalScene->getObjectByName(name);
			if(bone)
			{
				animRot[i] = bone->m_Rotation_Values;
				animTime[i] = bone->m_Rotation_Times;
			}
		}

		move += this->inDeltaTime()->getValue() * this->varSpeed()->getValue();

		float weight = 0;
		int keyFrame = -1;

		float currentFrameinAnim = fmod(move, inputTimeEnd - inputTimeStart) + inputTimeStart;

		for (size_t hingeId = 0; hingeId < animTime.size(); hingeId++)
		{
			if (!animTime[hingeId])
				continue;

			auto animTimeZ = animTime[hingeId][2];

			if (currentFrameinAnim < animTimeZ[0]) 
			{
				weight = -1;
				keyFrame = 1;
			}
			if (currentFrameinAnim > animTimeZ[animTimeZ.size()-1])
			{
				weight = -1;
				keyFrame = animTimeZ.size()-1;
			}

			for (size_t j = 0; j < animTimeZ.size(); j++)
			{
				if (currentFrameinAnim > animTimeZ[j] && currentFrameinAnim <= animTimeZ[j + 1])
				{
					float v = (currentFrameinAnim - animTimeZ[j]) / (animTimeZ[j + 1] - animTimeZ[j]);
					weight = v;
					keyFrame = j;
					break;
				}
			}	
		}



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



		if (keyFrame != -1) 
		{
			for (size_t hingeId = 0; hingeId < animRot.size(); hingeId++)
			{
				if (!animRot[hingeId])
					continue;
				auto animX = animRot[hingeId][0];
				auto animY = animRot[hingeId][1];
				auto animZ = animRot[hingeId][2];

				Real angle = 0;
				if (weight == -1){
					angle = animZ[keyFrame] * M_PI / 180 / angleDivede;
				}
				else {
					angle = lerp(animZ[keyFrame], animZ[keyFrame + 1], weight) * M_PI / 180 / angleDivede;
				}

				c_hinge[hingeId].setRange(angle, angle + epsilonAngle);
			}
		}


		d_hinge.assign(c_hinge);


	}


	
	DEFINE_CLASS(AnimationDriver);
}