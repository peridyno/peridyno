#pragma once
#include "Module/TopologyModule.h"
#include "VkDeviceArray.h"

namespace dyno
{
	class PointSet : public TopologyModule
	{
	public:
		PointSet();
		~PointSet() override;

	public:
		DArray<Vec3f> mPoints;
	};
}

