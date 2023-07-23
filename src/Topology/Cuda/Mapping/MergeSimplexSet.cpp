#include "MergeSimplexSet.h"

namespace dyno
{
	template<typename TDataType>
	MergeSimplexSet<TDataType>::MergeSimplexSet()
		: Node()
	{
		this->outSimplexSet()->allocate();
	}

	template<typename TDataType>
	MergeSimplexSet<TDataType>::~MergeSimplexSet()
	{

	}

	template<typename TDataType>
	void MergeSimplexSet<TDataType>::resetStates()
	{
		auto edge_set = this->inEdgeSet()->constDataPtr();
		auto tri_set = this->inTriangleSet()->constDataPtr();
		auto tet_set = this->inTetrahedronSet()->constDataPtr();

		auto& p0 = edge_set->getPoints();
		auto& p1 = tri_set->getPoints();
		auto& p2 = tet_set->getPoints();

		uint num = p0.size() + p1.size() + p2.size();

		auto simplices = this->outSimplexSet()->getDataPtr();

		DArray<Coord> points(num);
		
		points.assign(p0, p0.size(), 0, 0);
		points.assign(p1, p1.size(), 0, p0.size());
		points.assign(p2, p2.size(), 0, p0.size() + p1.size());

		simplices->setPoints(points);
		simplices->setSegments(edge_set->getEdges());
		simplices->setTriangles(tri_set->getTriangles());
		simplices->setTetrahedrons(tet_set->getTetrahedrons());

		simplices->update();
	}

	DEFINE_CLASS(MergeSimplexSet);
}