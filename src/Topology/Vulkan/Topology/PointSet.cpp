#include "PointSet.h"

namespace dyno
{
	PointSet::PointSet()
		: TopologyModule()
	{
	}

	PointSet::~PointSet()
	{
	}

	void PointSet::setPoints(std::vector<Vec3f>& points)
	{
		mPoints.assign(points);
	}

	void PointSet::setPoints(const DArray<Vec3f>& points)
	{
		mPoints.assign(points);
	}

	void PointSet::clear()
	{
		mPoints.clear();
	}

}