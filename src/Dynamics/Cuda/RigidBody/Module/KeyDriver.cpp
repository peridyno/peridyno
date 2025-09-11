#include "KeyDriver.h"

namespace dyno
{
	IMPLEMENT_TCLASS(KeyDriver, TDataType);

	template<typename TDataType>
	KeyDriver<TDataType>::KeyDriver()
		: KeyboardInputModule()
	{
		this->varCacheEvent()->setValue(false);
	}

	template<typename TDataType>
	void KeyDriver<TDataType>::onEvent(PKeyboardEvent event)
	{
		if (this->inReset()->getValue()) 
		{
			this->hingeAngle.clear();
			this->inReset()->setValue(false);
		}


		auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->inTopology()->getDataPtr());
		auto& d_hinge = topo->hingeJoints();
		CArray<HingeJoint> c_hinge;
		c_hinge.assign(d_hinge);

		Key2HingeConfig keyConfig = this->varHingeKeyConfig()->getValue();
		
		Real stepAngle = M_PI / 50;

		std::vector<HingeAction> currentHingeActions;

		auto key2HingeActionIterator = keyConfig.key2Hinge.find(event.key);

		if (key2HingeActionIterator != keyConfig.key2Hinge.end())
			currentHingeActions = key2HingeActionIterator->second;

		if (currentHingeActions.size())
		{
			for (auto action : currentHingeActions)
			{
				int keyJointID = action.joint;
				float keyValue = action.value;

				auto angleIterator = hingeAngle.find(keyJointID);
				if (angleIterator != hingeAngle.end())
					hingeAngle[keyJointID] = hingeAngle[keyJointID] + keyValue * stepAngle;
				else
					hingeAngle[keyJointID] = keyValue * stepAngle;

				c_hinge[keyJointID].setRange(hingeAngle[keyJointID], hingeAngle[keyJointID] + M_PI / 360);
			}
			
			
		}

		d_hinge.assign(c_hinge);


	}

	DEFINE_CLASS(KeyDriver);
}