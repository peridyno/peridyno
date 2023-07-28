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
		mEdges.clear();
		mVer2Edge.clear();
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
		counter.resize(this->mCoords.size());
		counter.reset();

		cuExecute(mEdges.size(),
			ES_CountEdges,
			counter,
			mEdges);

		mVer2Edge.resize(counter);

		counter.reset();
		cuExecute(mEdges.size(),
			ES_SetupEdgeIds,
			mVer2Edge,
			mEdges);

		counter.clear();

		return mVer2Edge;
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::requestPointNeighbors(DArrayList<int>& lists)
	{
		if (this->mCoords.isEmpty())
			return;

		DArray<uint> counts;
		counts.resize(this->mCoords.size());
		counts.reset();

		cuExecute(mEdges.size(),
			K_CountNumber,
			counts,
			mEdges);

		lists.resize(counts);

		cuExecute(mEdges.size(),
			K_StoreIds,
			lists,
			mEdges);

		counts.clear();
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::loadSmeshFile(std::string filename)
	{
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::copyFrom(EdgeSet<TDataType>& edgeSet)
	{
		mEdges.resize(edgeSet.mEdges.size());
		mEdges.assign(edgeSet.mEdges);

		mVer2Edge.assign(edgeSet.mVer2Edge);

		PointSet<TDataType>::copyFrom(edgeSet);
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::setEdges(std::vector<Edge>& edges)
	{
		mEdges.assign(edges);

		this->tagAsChanged();
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::setEdges(DArray<Edge>& edges)
	{
		mEdges.resize(edges.size());
		mEdges.assign(edges);

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
		return mEdges.size() == 0 && PointSet<TDataType>::isEmpty();
	}

	DEFINE_CLASS(EdgeSet);
}