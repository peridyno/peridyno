#include "CollisionDetectionBroadPhase.h"
#include <thrust/sort.h>

namespace dyno
{
	typedef unsigned long long int PKey;

	void print(DArray<int> arr)
	{
		CArray<int> h_arr;
		h_arr.resize(arr.size());

		h_arr.assign(arr);

		for (uint i = 0; i < h_arr.size(); i++)
		{
			printf("%d: %d \n", i, h_arr[i]);
		}

		h_arr.clear();
	};

	void print(DArray<PKey> arr)
	{
		CArray<PKey> h_arr;
		h_arr.resize(arr.size());

		h_arr.assign(arr);

		for (uint i = 0; i < h_arr.size(); i++)
		{
			int id = h_arr[i] & UINT_MAX;
			printf("%d: %d \n", i, id);
		}

		h_arr.clear();
	};

	IMPLEMENT_TCLASS(CollisionDetectionBroadPhase, TDataType)

		template<typename TDataType>
	CollisionDetectionBroadPhase<TDataType>::CollisionDetectionBroadPhase()
		: CollisionModel()
	{
		this->varGridSizeLimit()->setValue(0.01);
	}

	template<typename TDataType>
	CollisionDetectionBroadPhase<TDataType>::~CollisionDetectionBroadPhase()
	{
		octree.release();
	}

	template<typename Coord>
	__global__ void CDBP_SetupCorners(
		DArray<Coord> v0,
		DArray<Coord> v1,
		DArray<AABB> box)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= box.size()) return;

		v0[tId] = box[tId].v0;
		v1[tId] = box[tId].v1;
	}


	template<typename Real>
	__global__ void CDBP_ComputeAABBSize(
		DArray<Real> h,
		DArray<AABB> boundingBox)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= boundingBox.size()) return;

		AABB box = boundingBox[tId];

		h[tId] = max(box.v1[0] - box.v0[0], max(box.v1[1] - box.v0[1], box.v1[2] - box.v0[2]));
	}

	template<typename TDataType>
	__global__ void CDBP_RequestIntersectionNumber(
		DArray<int> count,
		DArray<AABB> boundingBox,
		SparseOctree<TDataType> octree)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		count[tId] = octree.requestIntersectionNumberFromBottom(boundingBox[tId]);
	}

	template<typename TDataType>
	__global__ void CDBP_RequestIntersectionIds(
		DArray<int> ids,
		DArray<int> count,
		DArray<AABB> boundingBox,
		SparseOctree<TDataType> octree)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		octree.reqeustIntersectionIdsFromBottom(ids.begin() + count[tId], boundingBox[tId]);
	}



	template<typename TDataType>
	__global__ void CDBP_RequestIntersectionNumber(
		DArray<int> count,
		DArray<AABB> boundingBox,
		SparseOctree<TDataType> octree,
		bool self_collision)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;
		if (!self_collision)
			count[tId] = octree.requestIntersectionNumberFromBottom(boundingBox[tId]);
		else
			count[tId] = octree.requestIntersectionNumberFromLevel(boundingBox[tId], octree.requestLevelNumber(boundingBox[tId]));
	}

	template<typename TDataType>
	__global__ void CDBP_RequestIntersectionIds(
		DArray<int> ids,
		DArray<int> count,
		DArray<AABB> boundingBox,
		SparseOctree<TDataType> octree,
		bool self_collision)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		if (!self_collision)
			octree.reqeustIntersectionIdsFromBottom(ids.begin() + count[tId], boundingBox[tId]);
		else
			octree.reqeustIntersectionIdsFromLevel(ids.begin() + count[tId], boundingBox[tId], octree.requestLevelNumber(boundingBox[tId]));
	}

	template<typename TDataType>
	__global__ void CDBP_RequestIntersectionNumberRemove(
		DArray<int> count,
		DArray<AABB> boundingBox_src,
		DArray<AABB> boundingBox_tar,
		SparseOctree<TDataType> octree,
		int self_collision)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox_src.size()) return;
		if (self_collision)
			count[tId] = octree.requestIntersectionNumberFromLevel(boundingBox_src[tId], boundingBox_tar.begin(), octree.requestLevelNumber(boundingBox_src[tId]));
		else
			count[tId] = octree.requestIntersectionNumberFromBottom(boundingBox_src[tId], boundingBox_tar.begin());
	}

	template<typename TDataType>
	__global__ void CDBP_RequestIntersectionIdsRemove(
		DArray<int> ids,
		DArray<int> count,
		DArray<AABB> boundingBox_src,
		DArray<AABB> boundingBox_tar,
		SparseOctree<TDataType> octree,
		int self_collision
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox_src.size()) return;
		if (self_collision)
			octree.reqeustIntersectionIdsFromLevel(ids.begin() + count[tId], boundingBox_src[tId], boundingBox_tar.begin(), octree.requestLevelNumber(boundingBox_src[tId]));
		else
			octree.reqeustIntersectionIdsFromBottom(ids.begin() + count[tId], boundingBox_src[tId], boundingBox_tar.begin());
	}

	__global__ void CDBP_SetupKeys(
		DArray<PKey> keys,
		DArray<int> ids,
		DArray<int> count)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		int shift = count[tId];
		int total_num = count.size();
		int n = tId == total_num - 1 ? ids.size() - shift : count[tId + 1] - shift;

		for (int i = 0; i < n; i++)
		{
			uint id = ids[shift + i];
			PKey key_hi = tId;
			PKey key_lo = id;
			keys[shift + i] = key_hi << 32 | key_lo;
		}
	}

	template<typename TDataType>
	__global__ void CDBP_CountDuplicativeIds(
		DArray<int> new_count,
		DArray<PKey> ids,
		DArray<int> count,
		DArray<AABB> boundingBox,
		SparseOctree<TDataType> octree)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		int total_num = boundingBox.size();
		int shift = count[tId];
		int n = tId == total_num - 1 ? ids.size() - count[total_num - 1] : count[tId + 1] - shift;

		int col_num = 0;

		for (int i = 0; i < n; i++)
		{
			uint B_id = ids[shift + i] & UINT_MAX;
			if (i == 0 || B_id != (ids[shift + i - 1] & UINT_MAX))
			{
				col_num++;
			}
		}

		new_count[tId] = col_num;
	}
	template<typename TDataType>
	__global__ void CDBP_CountDuplicativeIds(
		DArray<int> new_count,
		DArray<PKey> ids,
		DArray<int> count,
		DArray<AABB> boundingBox,
		SparseOctree<TDataType> octree,
		bool self_collision)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		int total_num = boundingBox.size();
		int shift = count[tId];
		int n = tId == total_num - 1 ? ids.size() - count[total_num - 1] : count[tId + 1] - shift;

		int col_num = 0;

		for (int i = 0; i < n; i++)
		{
			uint B_id = ids[shift + i] & UINT_MAX;
			if (i == 0 || B_id != (ids[shift + i - 1] & UINT_MAX))
			{
				if (self_collision)
				{
					if (B_id != tId)
					{
						if (octree.requestLevelNumber(boundingBox[tId]) == octree.requestLevelNumber(boundingBox[B_id]))
						{
							if (B_id > tId)
								col_num++;
						}
						else
							col_num++;
					}
				}
				else
					col_num++;
			}
		}

		new_count[tId] = col_num;
	}

	template<typename TDataType>
	__global__ void CDBP_RemoveDuplicativeIds(
		DArray<int> new_ids,
		DArray<int> new_count,
		DArray<PKey> ids,
		DArray<int> count,
		DArray<AABB> boundingBox,
		SparseOctree<TDataType> octree)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		int total_num = boundingBox.size();
		int shift = count[tId];
		int n = tId == total_num - 1 ? ids.size() - count[total_num - 1] : count[tId + 1] - shift;

		int col_num = 0;

		int shift_new = new_count[tId];

		for (int i = 0; i < n; i++)
		{
			uint B_id = ids[shift + i] & UINT_MAX;
			if (i == 0 || B_id != (ids[shift + i - 1] & UINT_MAX))
			{
				new_ids[shift_new + col_num] = B_id;
				col_num++;
			}
		}
	}

	template<typename TDataType>
	__global__ void CDBP_RemoveDuplicativeIds(
		DArrayList<int> contactLists,
		DArray<PKey> ids,
		DArray<int> count,
		DArray<AABB> boundingBox,
		SparseOctree<TDataType> octree,
		bool self_collision)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		int total_num = boundingBox.size();
		int shift = count[tId];
		int n = tId == total_num - 1 ? ids.size() - count[total_num - 1] : count[tId + 1] - shift;

		List<int>& cList_i = contactLists[tId];

		for (int i = 0; i < n; i++)
		{
			uint B_id = ids[shift + i] & UINT_MAX;
			if (i == 0 || B_id != (ids[shift + i - 1] & UINT_MAX))
			{
				if (self_collision)
				{
					if (B_id != tId)
					{

						if (octree.requestLevelNumber(boundingBox[tId]) == octree.requestLevelNumber(boundingBox[B_id]))
						{

							if (B_id > tId)
							{
								cList_i.insert(B_id);
							}
						}
						else
						{
							cList_i.insert(B_id);
						}
					}
				}
				else
				{
					cList_i.insert(B_id);
				}
			}
		}
	}


	__global__ void CDBP_RevertIds(
		DArray<int> elements)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= elements.size()) return;


		int id = elements[tId];
		elements[tId] = -id - 1;
	}

	template<typename TDataType>
	void CollisionDetectionBroadPhase<TDataType>::doCollision()
	{
		auto& aabb_src = this->inSource()->getData();
		auto& aabb_tar = this->inTarget()->getData();

		if (this->outContactList()->isEmpty()) {
			this->outContactList()->allocate();
		}
		
		auto& contacts = this->outContactList()->getData();

		DArray<Coord> v0_arr;
		DArray<Coord> v1_arr;

		DArray<Real> h_arr;

		v0_arr.resize(aabb_tar.size());
		v1_arr.resize(aabb_tar.size());
		h_arr.resize(aabb_tar.size());

		cuExecute(aabb_tar.size(),
			CDBP_SetupCorners,
			v0_arr,
			v1_arr,
			aabb_tar);

		cuExecute(aabb_tar.size(),
			CDBP_ComputeAABBSize,
			h_arr,
			aabb_tar);

		auto min_val = m_reduce_real.minimum(h_arr.begin(), h_arr.size());
		auto min_v0 = m_reduce_coord.minimum(v0_arr.begin(), v0_arr.size());
		auto max_v1 = m_reduce_coord.maximum(v1_arr.begin(), v1_arr.size());


		min_val = max(min_val, this->varGridSizeLimit()->getData());

		h_arr.clear();
		v0_arr.clear();
		v1_arr.clear();


		octree.setSpace(min_v0 - min_val, min_val, max(max_v1[0] - min_v0[0], max(max_v1[1] - min_v0[1], max_v1[2] - min_v0[2])) + 2.0f * min_val);
		//octree.setSpace(Coord(0,0,0) - min_val, min_val, 1.0f + 2.0f * min_val);

		octree.construct(aabb_tar);

		//if(octree.getLevelMax() > 9)
		//	octree.printPostOrderedTree();

		DArray<int> counter;
		counter.resize(aabb_src.size());
		/*
		cuExecute(aabb_src.size(),
			CDBP_RequestIntersectionNumber,
			counter,
			aabb_src,
			octree,
			self_collision);
		*/
		cuExecute(aabb_src.size(),
			CDBP_RequestIntersectionNumberRemove,
			counter,
			aabb_src,
			aabb_tar,
			octree,
			self_collision
		);

		int total_node_num = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		DArray<int> ids;
		ids.resize(total_node_num);
		/*
		cuExecute(aabb_src.size(),
			CDBP_RequestIntersectionIds,
			ids,
			counter,
			aabb_src,
			octree,
			self_collision);
			*/
		cuExecute(aabb_src.size(),
			CDBP_RequestIntersectionIdsRemove,
			ids,
			counter,
			aabb_src,
			aabb_tar,
			octree,
			self_collision);

		// 		print(counter);
		// 		print(ids);

		DArray<PKey> keys;
		keys.resize(ids.size());


		//remove duplicative ids and self id
		cuExecute(counter.size(),
			CDBP_SetupKeys,
			keys,
			ids,
			counter);

		thrust::sort(thrust::device, keys.begin(), keys.begin() + keys.size());

		// 		print(keys);
		// 		print(counter);

		DArray<int> new_count(counter.size());
		cuExecute(aabb_src.size(),
			CDBP_CountDuplicativeIds,
			new_count,
			keys,
			counter,
			aabb_src,
			octree,
			self_collision);

		contacts.resize(new_count);

// 		int ele_num = thrust::reduce(thrust::device, index.begin(), index.begin() + index.size(), (int)0, thrust::plus<int>());
// 		thrust::exclusive_scan(thrust::device, index.begin(), index.begin() + index.size(), index.begin());
// 
// 		elements.resize(ele_num);

		cuExecute(aabb_src.size(),
			CDBP_RemoveDuplicativeIds,
			contacts,
			keys,
			counter,
			aabb_src,
			octree,
			self_collision);

		/*
		cuExecute(elements.size(),
			CDBP_RevertIds,
			elements);
			*/
			//printf("FROM OCT: %d\n", elements.size());

		octree.release();
		ids.clear();
		keys.clear();
		counter.clear();
		new_count.clear();
	}

	DEFINE_CLASS(CollisionDetectionBroadPhase);
}