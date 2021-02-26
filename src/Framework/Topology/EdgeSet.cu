#include "EdgeSet.h"
#include <vector>
#include <Utility.h>
#include "Array/DynamicArray.h"

#include <thrust/sort.h>


namespace dyno
{
	template<typename TDataType>
	EdgeSet<TDataType>::EdgeSet()
	{
	}

	template<typename TDataType>
	EdgeSet<TDataType>::~EdgeSet()
	{
		

	}

	__global__ void K_CountNumber(
		DeviceArray<int> num,
		DeviceArray<TopologyModule::Edge> edges)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= edges.size()) return;

		TopologyModule::Edge edge = edges[eId];

		atomicAdd(&(num[edge[0]]), 1);
		atomicAdd(&(num[edge[1]]), 1);
	}

	__global__ void K_StoreIds(
		ListArray<int> ids,
		DeviceArray<TopologyModule::Edge> edges)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= edges.size()) return;

		TopologyModule::Edge edge = edges[eId];
		int v0 = edge[0];
		int v1 = edge[1];

		ids[v0].atomicInsert(v1);
		ids[v1].atomicInsert(v0);
	}
	

	template<typename TDataType>
	void EdgeSet<TDataType>::updatePointNeighbors()
	{
		if (this->m_coords.isEmpty())
			return;

		DeviceArray<int> counts;
		counts.resize(m_coords.size());
		counts.reset();
		
		cuExecute(m_edges.size(),
			K_CountNumber,
			counts,
			m_edges);

		m_pointNeighbors.allocate(counts);

		cuExecute(m_edges.size(),
			K_StoreIds,
			m_pointNeighbors,
			m_edges);

		counts.release();
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::loadSmeshFile(std::string filename)
	{
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::copyFrom(EdgeSet<TDataType>& edgeSet)
	{
		m_edges.resize(edgeSet.m_edges.size());
		Function1Pt::copy(m_edges, edgeSet.m_edges);

		PointSet<TDataType>::copyFrom(edgeSet);
	}

#ifdef PRECISION_FLOAT
	template class EdgeSet<DataType3f>;
#else
	template class EdgeSet<DataType3d>;
#endif
}