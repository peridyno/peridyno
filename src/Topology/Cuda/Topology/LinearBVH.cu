#include "LinearBVH.h"

#include "Object.h"
#include "STL/Stack.h"
#include "Math/SimpleMath.h"

#include <thrust/sort.h>

#include "Timer.h"

namespace dyno 
{
	template<typename TDataType>
	LinearBVH<TDataType>::LinearBVH()
	{
	}

	template<typename TDataType>
	LinearBVH<TDataType>::~LinearBVH()
	{
	}

	template<typename TDataType>
	void LinearBVH<TDataType>::release()
	{
		mAllNodes.clear();
		mCenters.clear();
		mSortedAABBs.clear();
		mSortedObjectIds.clear();
		mFlags.clear();		//Flags used for calculating bounding box
		mMortonCodes.clear();
	}

	template<typename Coord, typename AABB>
	__global__ void LBVH_CalculateCenter(
		DArray<Coord> center,
		DArray<AABB> aabb)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= aabb.size()) return;

		center[tId] = Real(0.5) * (aabb[tId].v0 + aabb[tId].v1);
	}

	// Expands a 10-bit integer into 30 bits
	// by inserting 2 zeros after each bit.
	__device__ uint expandBits(uint v)
	{
		v = (v * 0x00010001u) & 0xFF0000FFu;
		v = (v * 0x00000101u) & 0x0F00F00Fu;
		v = (v * 0x00000011u) & 0xC30C30C3u;
		v = (v * 0x00000005u) & 0x49249249u;
		return v;
	}

	// Calculates a 30-bit Morton code for the
	// given 3D point located within the unit cube [0,1].
	template<typename Real>
	__device__ uint morton3D(Real x, Real y, Real z)
	{
		x = min(max(x * Real(1024), Real(0)), Real(1023));
		y = min(max(y * Real(1024), Real(0)), Real(1023));
		z = min(max(z * Real(1024), Real(0)), Real(1023));
		uint xx = expandBits((uint)x);
		uint yy = expandBits((uint)y);
		uint zz = expandBits((uint)z);
		return xx * 4 + yy * 2 + zz;
	}

	template<typename Real, typename Coord>
	__global__ void LBVH_CalculateMortonCode(
		DArray<uint64> morton,
		DArray<uint> objectId,
		DArray<Coord> center,
		Coord orgin,
		Real L)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= center.size()) return;

		Coord scaled = (center[tId] - orgin) / L;

		uint64 m64 = morton3D(scaled.x, scaled.y, scaled.z);
		m64 <<= 32;
		m64 |= tId;

		//printf("morton %u: %u; %llu \n", tId, morton3D(scaled.x, scaled.y, scaled.z), m64);

		morton[tId] = m64;
		objectId[tId] = tId;
	}

	__device__ int findSplit(uint* sortedMortonCodes,
		int           first,
		int           last)
	{
		// Identical Morton codes => split the range in the middle.

		uint firstCode = sortedMortonCodes[first];
		uint lastCode = sortedMortonCodes[last];

		if (firstCode == lastCode)
			return (first + last) >> 1;

		// Calculate the number of highest bits that are the same
		// for all objects, using the count-leading-zeros intrinsic.

		int commonPrefix = __clz(firstCode ^ lastCode);

		// Use binary search to find where the next bit differs.
		// Specifically, we are looking for the highest object that
		// shares more than commonPrefix bits with the first one.

		int split = first; // initial guess
		int step = last - first;

		do
		{
			step = (step + 1) >> 1; // exponential decrease
			int newSplit = split + step; // proposed new position

			if (newSplit < last)
			{
				uint splitCode = sortedMortonCodes[newSplit];
				int splitPrefix = __clz(firstCode ^ splitCode);
				if (splitPrefix > commonPrefix)
					split = newSplit; // accept proposal
			}
		} while (step > 1);

		return split;
	}

	template<typename Node, typename AABB>
	__global__ void LBVH_ConstructBinaryRadixTree(
		DArray<Node> bvhNodes,
		DArray<AABB> sortedAABBs,
		DArray<AABB> aabbs,
		DArray<uint64> mortonCodes,
		DArray<uint> sortedObjectIds) 
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int N = sortedObjectIds.size();

		if (i >= N) return;

// 		printf("Num: %d \n", N);
// 
// 		printf("sorted morton %d: %llu \n", i, mortonCodes[i]);

		sortedAABBs[i + N - 1] = aabbs[sortedObjectIds[i]];

		if (i >= N - 1) return;

		//Calculate the length of the longest common prefix between i and j, note i should be in the range of [0, N-1]
		auto delta = [&](int _i, int _j) -> int {
			if (_j < 0 || _j >= N) return -1;
			return __clzll(mortonCodes[_i] ^ mortonCodes[_j]);
		};

//		printf("Test CLZ: %d \n", __clzll(mortonCodes[1]));

		int d = delta(i, i + 1) - delta(i, i - 1) > 0 ? 1 : -1;

// 		printf("%u %d \n", i, d);
// 
// 		printf("delta: %d \n", delta(0, 1));

		// Compute upper bound for the length of the range
		int delta_min = delta(i, i - d);

//		printf("delta_min %d %d: %d \n", i, i - d, delta_min);

		int len_max = 2;
		while (delta(i, i + len_max * d) > delta_min)
		{
			len_max *= 2;
		}

		// Find the other end using binary search
		int len = 0;
		for (int t = len_max / 2; t > 0; t = t / 2)
		{
			if (delta(i, i + (len + t) * d) > delta_min)
			{
				len = len + t;
			}
		}

		int j = i + len * d;

		// Find the split position using binary search
		int delta_node = delta(i, j);
		int s = 0;

// 		if (i == 41)
// 		{
// 			printf("len: %d \n", len);
// 		}

		for (int t = (len + 1) / 2; t > 0; t = t == 1 ? 0 : (t + 1) / 2)
		{
			if (delta(i, i + (s + t) * d) > delta_node)
			{
				s = s + t;
// 				if (i == 41)
// 				{
// 					printf("s: %d; t: %d \n", s, t);
// 				}
			}
		}
		int gamma = i + s * d + minimum(d, (int)0);

// 		printf("i-j: %d %d; Gamma: %d \n", i, j, gamma);
// 
// 		if (i == 41)
// 		{
// 			printf("21 22 23 24 dir: %d; %llu; %llu; %llu; %llu \n", d, mortonCodes[21], mortonCodes[22], mortonCodes[23], mortonCodes[24]);
// 			printf("0 41 42 dir: %llu; %llu; %llu \n", mortonCodes[0], mortonCodes[41], mortonCodes[42]);
// 		}

		//printf("Gamma: %u \n", gamma);

		//Output child pointers
		int left_idx = minimum(i, j) == gamma ? gamma + N - 1 : gamma;
		int right_idx = maximum(i, j) == gamma + 1 ? gamma + N : gamma + 1;

//		printf("i: %d, j: %d Left: %d; Right: %d; \n", i, j, left_idx, right_idx);

		bvhNodes[i].left = left_idx;
		bvhNodes[i].right = right_idx;

		bvhNodes[left_idx].parent = i;
		bvhNodes[right_idx].parent = i;
	}

	template<typename Node, typename AABB>
	__global__ void LBVH_CalculateBoundingBox(
		DArray<AABB> sortedAABBs,
		DArray<Node> bvhNodes,
		DArray<uint> flags)
	{
		uint i = threadIdx.x + (blockIdx.x * blockDim.x);
		uint N = flags.size();

		if (i >= N) return;
		
		//Output AABBs of leaf nodes
// 		auto v0 = sortedAABBs[i + N - 1].v0;
// 		auto v1 = sortedAABBs[i + N - 1].v1;
// 		printf("%d: idx, %f %f %f; %f %f %f \n", i + N - 1, v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);

		int idx = bvhNodes[i + N - 1].parent;
		while (idx != EMPTY) // means idx == 0
		{
			//printf("Left: %u; Right: %u, \n", idx->left->idx, idx->right->idx);
			const int old = atomicCAS(flags.begin() + idx, 0, 1);
			if (old == 0)
			{
				// this is the first thread entered here.
				// wait the other thread from the other child node.
				return;
			}
			assert(old == 1);
			// here, the flag has already been 1. it means that this
			// thread is the 2nd thread. merge AABB of both childlen.

			const int l_idx = bvhNodes[idx].left;
			const int r_idx = bvhNodes[idx].right;
			const AABB l_aabb = sortedAABBs[l_idx];
			const AABB r_aabb = sortedAABBs[r_idx];
			sortedAABBs[idx] = l_aabb.merge(r_aabb);

			//Output AABBs of internal nodes
// 			auto v0 = sortedAABBs[idx].v0;
// 			auto v1 = sortedAABBs[idx].v1;
// 			printf("%d: idx, %f %f %f; %f %f %f \n", idx, v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);

			// look the next parent...
			idx = bvhNodes[idx].parent;

			//printf("BB %d, \n", idx);
		}
	}

	template<typename Node>
	__global__ void LBVH_InitialAllNodes(
		DArray<Node> bvhNodes)
	{
		uint i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= bvhNodes.size()) return;

		bvhNodes[i] = Node();
	}


	template<typename TDataType>
	void LinearBVH<TDataType>::construct(DArray<AABB>& aabb)
	{
		uint num = aabb.size();

		if (mCenters.size() != num){
			mCenters.resize(num);
			mMortonCodes.resize(num);
			mSortedObjectIds.resize(num);
			mFlags.resize(num);

			mSortedAABBs.resize(2 * num - 1);
			mAllNodes.resize(2 * num - 1);
		}

		cuExecute(num,
			LBVH_CalculateCenter,
			mCenters,
			aabb);

		Reduction<Coord> mReduce;
		Coord v_min = mReduce.minimum(mCenters.begin(), mCenters.size());
		Coord v_max = mReduce.maximum(mCenters.begin(), mCenters.size());

		Real L = std::max(v_max[0] - v_min[0], std::max(v_max[1] - v_min[1], v_max[2] - v_min[2]));
		L = L < REAL_EPSILON ? Real(1) : L; //To avoid being divided by zero

		Coord origin = Real(0.5) * (v_min + v_max) - Real(0.5) * L;

		cuExecute(num,
			LBVH_CalculateMortonCode,
			mMortonCodes,
			mSortedObjectIds,
			mCenters,
			origin,
			L);

// 		GTimer timer;
// 		timer.start();
		thrust::sort_by_key(thrust::device, mMortonCodes.begin(), mMortonCodes.begin() + mMortonCodes.size(), mSortedObjectIds.begin());
// 		timer.stop();
// 		std::cout << "Sort: " << timer.getElapsedTime() << std::endl;

		cuExecute(mAllNodes.size(),
			LBVH_InitialAllNodes,
			mAllNodes);

//		timer.start();
		cuExecute(num,
			LBVH_ConstructBinaryRadixTree,
			mAllNodes,
			mSortedAABBs,
			aabb,
			mMortonCodes,
			mSortedObjectIds);
// 		timer.stop();
// 		std::cout << "Construct: " << timer.getElapsedTime() << std::endl;

// 		CArray<Node> hArray;
// 		hArray.assign(mAllNodes);

//		timer.start();
		mFlags.reset();
		cuExecute(num,
			LBVH_CalculateBoundingBox,
			mSortedAABBs,
			mAllNodes,
			mFlags);
// 		timer.stop();
// 		std::cout << "BoundingBox: " << timer.getElapsedTime() << std::endl;
	}

	template<typename TDataType>
	GPU_FUNC uint LinearBVH<TDataType>::requestIntersectionNumber(const AABB& queryAABB, const int queryId) const
	{
		// Allocate traversal stack from thread-local memory,
		// and push NULL to indicate that there are no postponed nodes.
		int buffer[64];

		Stack<int> stack;
		stack.reserve(buffer, 64);

		uint N = mSortedObjectIds.size();

		// Traverse nodes starting from the root.
		uint ret = 0;
		int idx = 0;
		do 
		{
			// Check each child node for overlap.
			int idxL = mAllNodes[idx].left;
			int idxR = mAllNodes[idx].right;
			bool overlapL = queryAABB.checkOverlap(getAABB(idxL));
			bool overlapR = queryAABB.checkOverlap(getAABB(idxR));

			// Query overlaps a leaf node => report collision.
			if (overlapL && mAllNodes[idxL].isLeaf()) {
				int objId = mSortedObjectIds[idxL - N + 1];
				if(objId > queryId) ret++;
			}
			
			if (overlapR && mAllNodes[idxR].isLeaf()) {
				int objId = mSortedObjectIds[idxR - N + 1];
				if (objId > queryId) ret++;
			}
			
			// Query overlaps an internal node => traverse.
			bool traverseL = (overlapL && !mAllNodes[idxL].isLeaf());
			bool traverseR = (overlapR && !mAllNodes[idxR].isLeaf());

			if (!traverseL && !traverseR) {
				idx = !stack.empty() ? stack.top() : EMPTY; // pop
				stack.pop();
			}
			else
			{
				idx = (traverseL) ? idxL : idxR;
				if (traverseL && traverseR)
					stack.push(idxR); // push
			}
		} while (idx != EMPTY);

		return ret;
	}

	template<typename TDataType>
	GPU_FUNC void LinearBVH<TDataType>::requestIntersectionIds(List<int>& ids, const AABB& queryAABB, const int queryId) const
	{
		// Allocate traversal stack from thread-local memory,
		// and push NULL to indicate that there are no postponed nodes.
		int buffer[64];

		Stack<int> stack;
		stack.reserve(buffer, 64);

		uint N = mSortedObjectIds.size();

		// Traverse nodes starting from the root.
		uint ret = 0;
		int idx = 0;
		do
		{
			// Check each child node for overlap.
			int idxL = mAllNodes[idx].left;
			int idxR = mAllNodes[idx].right;
			bool overlapL = queryAABB.checkOverlap(getAABB(idxL));
			bool overlapR = queryAABB.checkOverlap(getAABB(idxR));

			// Query overlaps a leaf node => report collision.
			if (overlapL && mAllNodes[idxL].isLeaf()) {
				int objId = mSortedObjectIds[idxL - N + 1];
				if (objId > queryId) 
					ids.insert(objId);
			}

			if (overlapR && mAllNodes[idxR].isLeaf()) {
				int objId = mSortedObjectIds[idxR - N + 1];
				if (objId > queryId) 
					ids.insert(objId);
			}

			// Query overlaps an internal node => traverse.
			bool traverseL = (overlapL && !mAllNodes[idxL].isLeaf());
			bool traverseR = (overlapR && !mAllNodes[idxR].isLeaf());

			if (!traverseL && !traverseR) {
				idx = !stack.empty() ? stack.top() : EMPTY; // pop
				stack.pop();
			}
			else
			{
				idx = (traverseL) ? idxL : idxR;
				if (traverseL && traverseR)
					stack.push(idxR); // push
			}
		} while (idx != EMPTY);
	}

	DEFINE_CLASS(LinearBVH);
}