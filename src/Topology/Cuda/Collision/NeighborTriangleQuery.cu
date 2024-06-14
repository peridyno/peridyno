#include "NeighborTriangleQuery.h"

#include "Topology/SparseOctree.h"

#include "Collision/CollisionDetectionBroadPhase.h"

namespace dyno
{
	IMPLEMENT_TCLASS(NeighborTriangleQuery, TDataType)

	template<typename TDataType>
	NeighborTriangleQuery<TDataType>::NeighborTriangleQuery()
		: ComputeModule()
	{
		mBroadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
	}

	template<typename TDataType>
	NeighborTriangleQuery<TDataType>::~NeighborTriangleQuery()
	{
	}

	template<typename Real, typename Coord>
	__global__ void NTQ_SetupAABB(
		DArray<AABB> boundingBox,
		DArray<Coord> position,
		Real radius)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		AABB box;
		Coord p = position[pId];
		box.v0 = p - radius;
		box.v1 = p + radius;

		boundingBox[pId] = box;
	}

	template<typename Coord>
	__global__ void NTQ_SetupAABB(
		DArray<AABB> boundingBox,
		DArray<Coord> vertex,
		DArray<TopologyModule::Triangle> tIndex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		AABB box;
		TopologyModule::Triangle index = tIndex[tId];

		Coord v0 = vertex[index[0]];
		Coord v1 = vertex[index[1]];
		Coord v2 = vertex[index[2]];

		box.v0 = minimum(v0, minimum(v1, v2));
		box.v1 = maximum(v0, maximum(v1, v2));

		boundingBox[tId] = box;
	}

	template<typename Coord>
	__global__ void NTQ_Narrow_Count(
		DArrayList<int> nbr,
		DArray<Coord> position,
		DArray<Coord> vertex,
		DArray<TopologyModule::Triangle> triangles,
		DArray<uint> count,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= position.size()) return;
		uint cnt = 0;
		
		List<int>& nbrIds_i = nbr[tId];
		int nbSize = nbrIds_i.size();  
		//printf("CNT_OKKKK nbSize: %d\n", nbSize);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbrIds_i[ne];
			Point3D point(position[tId]);
			Triangle3D t3d_n(vertex[triangles[j][0]], vertex[triangles[j][1]], vertex[triangles[j][2]]);
			Real p_dis_t = abs(point.distance(t3d_n));
			if (p_dis_t< radius && p_dis_t > EPSILON)
			{
				cnt++;
			}
		}
		count[tId] = cnt;
	}

	template<typename Coord>
	__global__ void NTQ_Narrow_Set(
		DArrayList<int> nbr,
		DArrayList<int> nbr_out,
		DArray<Coord> position,
		DArray<Coord> vertex,
		DArray<TopologyModule::Triangle> triangles,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= position.size()) return;
		int cnt = 0;

		List<int>& nbrIds_i = nbr[tId];
		int nbSize = nbrIds_i.size();  //int nbSize = nbr.getNeighborSize(tId); 
		List<int>& nbrOutIds_i = nbr_out[tId];
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbrIds_i[ne];

			Point3D point(position[tId]);
			Triangle3D t3d_n(vertex[triangles[j][0]], vertex[triangles[j][1]], vertex[triangles[j][2]]);
			Real proj_dist = abs(point.distance(t3d_n));

			if (proj_dist < radius && proj_dist > EPSILON)
			{
				nbrOutIds_i.insert(j);
			}
		}
	}

	template<typename TDataType>
	void NeighborTriangleQuery<TDataType>::compute()
	{
		int pNum = this->inPosition()->size();
		if (pNum == 0) return;

		if (mQueryAABB.size() != pNum) {
			mQueryAABB.resize(pNum);
		}

		auto ts = this->inTriangleSet()->constDataPtr();
		auto& triVertex = ts->getPoints();
		auto& triIndex = ts->getTriangles();

		int tNum = triIndex.size();
		if (tNum == 0) return;
		if (mQueriedAABB.size() != tNum) {
			mQueriedAABB.resize(tNum);
		}

		cuExecute(pNum,
			NTQ_SetupAABB,
			mQueryAABB,
			this->inPosition()->constData(),
			this->inRadius()->getValue() * 0.9);

		cuExecute(tNum,
			NTQ_SetupAABB,
			mQueriedAABB,
			triVertex,
			triIndex);

		Real radius = this->inRadius()->getValue();

		mBroadPhaseCD->varGridSizeLimit()->setValue(2 * radius);
		mBroadPhaseCD->inSource()->assign(mQueryAABB);

		if (this->inTriangleSet()->isModified()) {
			mBroadPhaseCD->inTarget()->assign(mQueriedAABB);
		}
		

		auto type = this->varSpatial()->getDataPtr()->currentKey();

		switch (type)
		{
		case Spatial::BVH:
			mBroadPhaseCD->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::BVH);
			break;
		case Spatial::OCTREE:
			mBroadPhaseCD->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::Octree);
			break;
		default:
			break;
		}

		mBroadPhaseCD->update();

		auto& nbr = mBroadPhaseCD->outContactList()->getData();

		if (this->outNeighborIds()->size() != pNum)
		{
			this->outNeighborIds()->allocate();
			DArray<uint> nbrNum;
			nbrNum.resize(pNum);
			nbrNum.reset();
			auto& nbrIds = this->outNeighborIds()->getData();
			nbrIds.resize(nbrNum);
			nbrNum.clear();

			//this->outNeighborIds()->getData().resize(p_num);
		}
		auto& nbrIds = this->outNeighborIds()->getData();

		//new
		DArray<uint> nbrNum;
		nbrNum.resize(pNum);
		nbrNum.reset();
		int sum1 = nbr.elementSize();
		if (sum1 > 0)
		{
			//printf("one %d %d\n", nbrNum.size(), p_num);
			cuExecute(pNum,
				NTQ_Narrow_Count,
				nbr,
				this->inPosition()->getData(),
				triVertex,
				triIndex,
				nbrNum,
				this->inRadius()->getData() * 0.9);

			nbrIds.resize(nbrNum);

			int sum = mReduce.accumulate(nbrNum.begin(), nbrNum.size());
			if (sum > 0)
			{
				cuExecute(pNum,
					NTQ_Narrow_Set,
					nbr,
					nbrIds,
					this->inPosition()->getData(),
					triVertex,
					triIndex,
					this->inRadius()->getData() * 0.9);

				nbrNum.clear();
			}
		}
	}

	DEFINE_CLASS(NeighborTriangleQuery);
}