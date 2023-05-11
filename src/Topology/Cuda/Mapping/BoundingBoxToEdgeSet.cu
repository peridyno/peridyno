#include "BoundingBoxToEdgeSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(BoundingBoxToEdgeSet, TDataType)

	template<typename TDataType>
	BoundingBoxToEdgeSet<TDataType>::BoundingBoxToEdgeSet()
		: TopologyMapping()
	{
	}

	template<typename Coord, typename AABB>
	__global__ void BBSS_SetupEdgeSet(
		DArray<Coord> vertices,
		DArray<TopologyModule::Edge> edges,
		DArray<AABB> aabbs)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= aabbs.size()) return;

		auto aabb = aabbs[tId];
		Coord v0 = aabb.v0;
		Coord v1 = aabb.v1;

		int id0 = 8 * tId;
		int id1 = 8 * tId + 1;
		int id2 = 8 * tId + 2;
		int id3 = 8 * tId + 3;
		int id4 = 8 * tId + 4;
		int id5 = 8 * tId + 5;
		int id6 = 8 * tId + 6;
		int id7 = 8 * tId + 7;

		vertices[id0] = v0;
		vertices[id1] = Coord(v0.x, v0.y, v1.z);
		vertices[id2] = Coord(v1.x, v0.y, v1.z);
		vertices[id3] = Coord(v1.x, v0.y, v0.z);
		vertices[id4] = Coord(v0.x, v1.y, v0.z);
		vertices[id5] = Coord(v0.x, v1.y, v1.z);
		vertices[id6] = Coord(v1.x, v1.y, v1.z);
		vertices[id7] = Coord(v1.x, v1.y, v0.z);

		edges[12 * tId] = TopologyModule::Edge(id0, id1);
		edges[12 * tId + 1] = TopologyModule::Edge(id1, id2);
		edges[12 * tId + 2] = TopologyModule::Edge(id2, id3);
		edges[12 * tId + 3] = TopologyModule::Edge(id3, id0);
		edges[12 * tId + 4] = TopologyModule::Edge(id0, id4);
		edges[12 * tId + 5] = TopologyModule::Edge(id1, id5);
		edges[12 * tId + 6] = TopologyModule::Edge(id2, id6);
		edges[12 * tId + 7] = TopologyModule::Edge(id3, id7);
		edges[12 * tId + 8] = TopologyModule::Edge(id4, id5);
		edges[12 * tId + 9] = TopologyModule::Edge(id5, id6);
		edges[12 * tId + 10] = TopologyModule::Edge(id6, id7);
		edges[12 * tId + 11] = TopologyModule::Edge(id7, id4);
	}

	template<typename TDataType>
	bool BoundingBoxToEdgeSet<TDataType>::apply()
	{
		if (this->outEdgeSet()->isEmpty())
			this->outEdgeSet()->allocate();

		auto& aabbs = this->inAABB()->getData();
		auto outSet = this->outEdgeSet()->getDataPtr();

		auto& vertices = outSet->getPoints();
		auto& edges = outSet->getEdges();

		uint num = aabbs.size();
		vertices.resize(8 * num);
		edges.resize(12 * num);

		cuExecute(num,
			BBSS_SetupEdgeSet,
			vertices,
			edges,
			aabbs);

		return true;
	}

	DEFINE_CLASS(BoundingBoxToEdgeSet);
}