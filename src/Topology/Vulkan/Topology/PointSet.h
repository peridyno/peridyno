#pragma once
#include "Module/TopologyModule.h"
#include "VkDeviceArray.h"
#include "Particle.h"

namespace dyno
{
	class PointSet : public dyno::TopologyModule
	{
	public:
		PointSet();
		~PointSet() override;

	public:
		VkDeviceArray<dyno::Vec3f> mPoints;
	};
}

