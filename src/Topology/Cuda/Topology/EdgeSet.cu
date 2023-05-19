#include "EdgeSet.h"
#include <vector>
#include "Array/ArrayList.h"

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
		m_edges.clear();
		m_ver2Edge.clear();
	}

	__global__ void K_CountNumber(
		DArray<uint> num,
		DArray<TopologyModule::Edge> edges)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= edges.size()) return;

		TopologyModule::Edge edge = edges[eId];

		atomicAdd(&(num[edge[0]]), 1);
		atomicAdd(&(num[edge[1]]), 1);
	}

	__global__ void K_StoreIds(
		DArrayList<int> ids,
		DArray<TopologyModule::Edge> edges)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= edges.size()) return;

		TopologyModule::Edge edge = edges[eId];
		int v0 = edge[0];
		int v1 = edge[1];

		ids[v0].atomicInsert(v1);
		ids[v1].atomicInsert(v0);
	}
	
	template<typename Edge>
	__global__ void ES_CountEdges(
		DArray<uint> counter,
		DArray<Edge> edges)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= edges.size()) return;

		Edge t = edges[tId];

		atomicAdd(&counter[t[0]], 1);
		atomicAdd(&counter[t[1]], 1);
	}

	template<typename Edge>
	__global__ void ES_SetupEdgeIds(
		DArrayList<int> edgeIds,
		DArray<Edge> edges)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= edges.size()) return;

		Edge t = edges[tId];

		edgeIds[t[0]].atomicInsert(tId);
		edgeIds[t[1]].atomicInsert(tId);
	}

	template<typename TDataType>
	DArrayList<int>& EdgeSet<TDataType>::getVer2Edge()
	{
		DArray<uint> counter;
		counter.resize(this->m_coords.size());
		counter.reset();

		cuExecute(m_edges.size(),
			ES_CountEdges,
			counter,
			m_edges);

		m_ver2Edge.resize(counter);

		counter.reset();
		cuExecute(m_edges.size(),
			ES_SetupEdgeIds,
			m_ver2Edge,
			m_edges);

		counter.clear();

		return m_ver2Edge;
	}


	template<typename TDataType>
	void EdgeSet<TDataType>::updatePointNeighbors()
	{
		if (this->m_coords.isEmpty())
			return;

		DArray<uint> counts;
		counts.resize(this->m_coords.size());
		counts.reset();
		
		cuExecute(m_edges.size(),
			K_CountNumber,
			counts,
			m_edges);

		this->m_pointNeighbors.resize(counts);

		cuExecute(m_edges.size(),
			K_StoreIds,
			this->m_pointNeighbors,
			m_edges);

		counts.clear();
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::loadSmeshFile(std::string filename)
	{
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::copyFrom(EdgeSet<TDataType>& edgeSet)
	{
		m_edges.resize(edgeSet.m_edges.size());
		m_edges.assign(edgeSet.m_edges);

		m_ver2Edge.assign(edgeSet.m_ver2Edge);

		PointSet<TDataType>::copyFrom(edgeSet);
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::setEdges(std::vector<Edge>& edges)
	{
		m_edges.assign(edges);

		this->tagAsChanged();
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::setEdges(DArray<Edge>& edges)
	{
		m_edges.resize(edges.size());
		m_edges.assign(edges);

		this->tagAsChanged();
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::updateTopology()
	{
		this->updateEdges();

		PointSet<TDataType>::updateTopology();
	}

	template<typename TDataType>
	bool EdgeSet<TDataType>::isEmpty()
	{
		return m_edges.size() == 0 && PointSet<TDataType>::isEmpty();
	}

	DEFINE_CLASS(EdgeSet);
}