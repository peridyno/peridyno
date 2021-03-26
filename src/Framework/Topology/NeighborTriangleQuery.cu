#include "NeighborTriangleQuery.h"
#include "Collision/CollisionDetectionBroadPhase.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(NeighborTriangleQuery, TDataType)

	template<typename TDataType>
	NeighborTriangleQuery<TDataType>::NeighborTriangleQuery()
		: ComputeModule()
	{
		this->inRadius()->setValue(Real(0.011));

		m_broadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
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
		DArray<TriangleIndex> tIndex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tIndex.size()) return;

		AABB box;
		TriangleIndex index = tIndex[tId];
		
		Coord v0 = vertex[index[0]];
		Coord v1 = vertex[index[1]];
		Coord v2 = vertex[index[2]];

		box.v0 = minimum(v0, minimum(v1, v2));
		box.v1 = maximum(v0, maximum(v1, v2));

		/*
		printf("Triangle AABB: %.3lf %.3lf %.3lf ++  %.3lf %.3lf %.3lf \n",
			box.v0[0], box.v0[1], box.v0[2],
			box.v1[0], box.v1[1], box.v1[2]);
			*/
		boundingBox[tId] = box;
	}


	template<typename TDataType>
	bool NeighborTriangleQuery<TDataType>::initializeImpl()
	{
		compute();
		return true;
	}
	 
	template<typename TDataType>
	void NeighborTriangleQuery<TDataType>::compute()
	{
		if (!this->inPosition()->isEmpty() && !this->inTriangleVertex()->isEmpty() && !this->inTriangleIndex()->isEmpty())
		{

			int p_num = this->inPosition()->getElementCount();
			if (m_queryAABB.size() != p_num)
			{
				m_queryAABB.resize(p_num);
			}

			int t_num = this->inTriangleIndex()->getElementCount();
			if (m_queriedAABB.size() != t_num)
			{
				m_queriedAABB.resize(t_num);
			}

			cuExecute(p_num,
				NTQ_SetupAABB,
				m_queryAABB,
				this->inPosition()->getValue(),
				this->inRadius()->getValue()*0.9);

			cuExecute(t_num,
				NTQ_SetupAABB,
				m_queriedAABB,
				this->inTriangleVertex()->getValue(),
				this->inTriangleIndex()->getValue());

			Real radius = this->inRadius()->getValue();

			m_broadPhaseCD->varGridSizeLimit()->setValue(2 * radius);



			if (this->outNeighborhood()->getElementCount() != p_num)
			{
				this->outNeighborhood()->setElementCount(p_num);
			}

			m_broadPhaseCD->inSource()->setValue(m_queryAABB);
			m_broadPhaseCD->inTarget()->setValue(m_queriedAABB);
			// 
			m_broadPhaseCD->update();


			m_broadPhaseCD->outContactList()->connect(this->outNeighborhood());
		}
	}

}