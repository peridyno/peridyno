#include "NeighborTriangleQuery.h"
#include "Primitive/Primitive3D.h"

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

		int tNum = this->inTriangles()->size();
		if (tNum == 0) return;
		if (mQueriedAABB.size() != tNum) {
			mQueriedAABB.resize(tNum);
		}

		cuExecute(pNum,
			NTQ_SetupAABB,
			mQueryAABB,
			this->inPosition()->getData(),
			this->inRadius()->getData() * 0.9);

		cuExecute(tNum,
			NTQ_SetupAABB,
			mQueriedAABB,
			this->inTriPosition()->getData(),
			this->inTriangles()->getData());

		Real radius = this->inRadius()->getData();

		mBroadPhaseCD->varGridSizeLimit()->setValue(2 * radius);
		mBroadPhaseCD->inSource()->assign(mQueryAABB);
		mBroadPhaseCD->inTarget()->assign(mQueriedAABB);

		auto type = this->varSpatial()->getDataPtr()->currentKey();

		switch (type)
		{
		case Spatial::BVH:
			mBroadPhaseCD->varAccelerationStructure()->getDataPtr()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::BVH);
		case Spatial::OCTREE:
			mBroadPhaseCD->varAccelerationStructure()->getDataPtr()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::Octree);
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
				this->inTriPosition()->getData(),
				this->inTriangles()->getData(),
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
					this->inTriPosition()->getData(),
					this->inTriangles()->getData(),
					this->inRadius()->getData() * 0.9);

				nbrNum.clear();
			}
		}
	}

	DEFINE_CLASS(NeighborTriangleQuery);
}