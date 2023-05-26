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

		DArray<Vec3f>& getPoints() { return mPoints; }

	public:
		DArray<Vec3f> mPoints;
	};
}

