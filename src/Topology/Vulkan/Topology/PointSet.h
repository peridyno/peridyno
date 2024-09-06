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

		void setPoints(std::vector<Vec3f>& points);
		void setPoints(const DArray<Vec3f>& points);

		void clear();

	public:
		DArray<Vec3f> mPoints;
	};
}

