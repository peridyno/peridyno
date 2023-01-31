#include "PointSet.h"

namespace dyno
{
	PointSet::PointSet()
		: TopologyModule()
	{
		this->setUpdateAlways(true);
	}

	PointSet::~PointSet()
	{
	}
}