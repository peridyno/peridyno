#include "SparseNeighborPointQuery.h"

#include <thrust/sort.h>

namespace dyno
{
	IMPLEMENT_TCLASS(SparseNeighborPointQuery, TDataType)

		template<typename TDataType>
	SparseNeighborPointQuery<TDataType>::SparseNeighborPointQuery()
		: ComputeModule()
	{
	}

	template<typename TDataType>
	SparseNeighborPointQuery<TDataType>::~SparseNeighborPointQuery()
	{
		octree.release();
	}

	template<typename Coord, typename TDataType>
	__global__ void CDBP_RequestIntersectionNumber(
		DArray<int> count,
		DArray<Coord> points,
		Real radius,
		SparseOctree<TDataType> octree)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		Coord p = points[tId];

		AABB aabb;
		aabb.v0 = p - radius;
		aabb.v1 = p + radius;

		count[tId] = octree.requestIntersectionNumberFromBottom(aabb);
	}

	template<typename Coord, typename TDataType>
	__global__ void CDBP_RequestIntersectionIds(
		DArrayList<int> lists,
		DArray<int> ids,
		DArray<int> count,
		DArray<Coord> points,
		Real radius,
		SparseOctree<TDataType> octree)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		Coord p = points[tId];
		int total_num = count.size();

		AABB aabb;
		aabb.v0 = p - radius;
		aabb.v1 = p + radius;

		octree.reqeustIntersectionIdsFromBottom(ids.begin() + count[tId], aabb);

		int n = tId == total_num - 1 ? ids.size() - count[total_num - 1] : count[tId + 1] - count[tId];

		List<int>& list_i = lists[tId];

		for (int t = 0; t < n; t++)
		{
			list_i.insert(ids[count[tId] + t]);
		}
	}

	template<typename Real, typename Coord>
	__global__ void CDBP_RequestNeighborSize(
		DArray<int> counter,
		DArray<Coord> srcPoints,
		DArray<Coord> tarPoints,
		DArrayList<int> lists,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counter.size()) return;

		Coord p_i = srcPoints[tId];

		List<int>& list_i = lists[tId];
		int nbSize = list_i.size();
		int num = 0;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (p_i - tarPoints[j]).norm();

			if (r < radius)
				num++;
// 
// 			if (tId == 0)
// 			{
// 				Coord p_j = tarPoints[j];
// 				printf("i: %f %f %f; j %f %f %f \n", p_i.x, p_i.y, p_i.z, p_j.x, p_j.y, p_j.z);
// 			}
		}

		//printf("i: %d \n", nbSize);

		counter[tId] = num;
	}

	template<typename Real, typename Coord>
	__global__ void CDBP_RequestNeighborIds(
		DArrayList<int> neighbors,
		DArray<Coord> srcPoints,
		DArray<Coord> tarPoints,
		DArrayList<int> lists,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= neighbors.size()) return;

		Coord p_i = srcPoints[tId];

		List<int>& list_i = lists[tId];
		int nbSize = list_i.size();

		List<int>& neList_i = neighbors[tId];
		int num = 0;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (p_i - tarPoints[j]).norm();

			if (r < radius)
			{
				neList_i.insert(j);
			}
		}
	}

	template<typename TDataType>
	void SparseNeighborPointQuery<TDataType>::compute()
	{
		auto& pSrc = this->inSource()->getData();
		auto& pTar = this->inTarget()->getData();

		uint numSrc = pSrc.size();
		uint numTar = pTar.size();

		if (this->outNeighborIds()->isEmpty()) {
			this->outNeighborIds()->allocate();
		}
		
		auto& neighborLists = this->outNeighborIds()->getData();

		auto min_v0 = m_reduce_coord.minimum(pTar.begin(), pTar.size());
		auto max_v1 = m_reduce_coord.maximum(pTar.begin(), pTar.size());

		octree.setSpace(min_v0, this->inRadius()->getData(), maximum(max_v1[0] - min_v0[0], maximum(max_v1[1] - min_v0[1], max_v1[2] - min_v0[2])));
		octree.construct(pTar, 0);

		DArray<int> counter(numSrc);

		cuExecute(numSrc,
			CDBP_RequestIntersectionNumber,
			counter,
			pSrc,
			this->inRadius()->getData(),
			octree);

		DArrayList<int> lists;
		lists.resize(counter);

		int total_num = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		DArray<int> ids(total_num);
		cuExecute(numSrc,
			CDBP_RequestIntersectionIds,
			lists,
			ids,
			counter,
			pSrc,
			this->inRadius()->getData(),
			octree);

		DArray<int> neighbor_counter(numSrc);
		cuExecute(numSrc,
			CDBP_RequestNeighborSize,
			neighbor_counter,
			pSrc,
			pTar,
			lists,
			this->inRadius()->getData()
		);

		neighborLists.resize(neighbor_counter);

		cuExecute(numSrc,
			CDBP_RequestNeighborIds,
			neighborLists,
			pSrc,
			pTar,
			lists,
			this->inRadius()->getData());

		counter.clear();
		lists.clear();
		ids.clear();
		neighbor_counter.clear();
	}

	DEFINE_CLASS(SparseNeighborPointQuery);
}