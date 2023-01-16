#include "EdgeSet.h"

namespace dyno
{
	EdgeSet::EdgeSet()
		: PointSet()
	{
	}

	EdgeSet::~EdgeSet()
	{
	}

	void EdgeSet::updateTopology()
	{
		this->updateEdges();
	}

}