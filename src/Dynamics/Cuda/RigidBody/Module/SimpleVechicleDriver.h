#pragma once
#include "Module/ComputeModule.h"

namespace dyno 
{
	class SimpleVechicleDriver : virtual public ComputeModule
	{
		DECLARE_CLASS(Vechicle)
	public:
		SimpleVechicleDriver();
		~SimpleVechicleDriver() override;

	public:

		DEF_VAR_IN(uint, FrameNumber, "Texture mesh of the vechicle");

		DEF_ARRAYLIST_IN(Transform3f, InstanceTransform, DeviceType::GPU, "Instance transforms");

	protected:
		void compute() override;

	protected:
		float theta = 0;
	};
}
