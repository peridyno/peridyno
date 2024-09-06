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

	void EdgeSet::setEdges(const DArray<Edge>& edges)
	{
		mEdgeIndex.assign(edges);
	}

	void EdgeSet::setEdges(const std::vector<Edge>& edges)
	{
		mEdgeIndex.assign(edges);
	}

	void EdgeSet::copyFrom(EdgeSet& es)
	{
		mEdgeIndex.assign(es.mEdgeIndex);
		mPoints.assign(es.mPoints);
	}

	void EdgeSet::updateTopology()
	{
		this->updateEdges();
	}

}