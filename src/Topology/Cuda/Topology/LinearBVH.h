/**
 * Copyright 2017-2023 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "DataTypes.h"
#include "TopologyConstants.h"

#include "Array/Array.h"
#include "STL/List.h"

#include "Primitive/Primitive3D.h"

#include "Algorithm/Reduction.h"

/**
 * @brief This class implements the lienar BVH based on "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees" 
 *			by Tero Karras, High Performance Graphics, 2012
 */
namespace dyno 
{
	class BVHNode
	{
	public:
		DYN_FUNC BVHNode() {
			parent = EMPTY;
			left = EMPTY;
			right = EMPTY;
		};

		DYN_FUNC bool isLeaf() { return left == EMPTY && right == EMPTY; }

		int parent;

		int left;
		int right;
	};

	template<typename TDataType>
	class LinearBVH
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename ::dyno::TAlignedBox3D<Real> AABB;
		typedef typename ::dyno::BVHNode Node;
		typedef typename ::dyno::BVHNode* NodePtr;

		LinearBVH();
		~LinearBVH();

		void construct(DArray<AABB>& aabb);

		GPU_FUNC uint requestIntersectionNumber(const AABB& queryAABB, const int queryId = EMPTY) const;
		GPU_FUNC void requestIntersectionIds(List<int>& ids, const AABB& queryAABB, const int queryId = EMPTY) const;

		GPU_FUNC NodePtr getRoot() const { return &mAllNodes[0]; }

		GPU_FUNC AABB getAABB(const uint idx) const { return mSortedAABBs[idx]; }
		GPU_FUNC uint getObjectIdx(const uint idx) const { return mSortedObjectIds[idx]; }

		CPU_FUNC DArray<AABB>& getSortedAABBs() { return mSortedAABBs; }

		/**
		 * @brief Call release() to release allocated memory explicitly, do not call this function from the decontructor.
		 *
		 */
		void release();

	private:
		DArray<Node> mAllNodes;

		DArray<Coord> mCenters;		//AABB center

		DArray<AABB> mSortedAABBs;
		DArray<uint> mSortedObjectIds;

		DArray<uint> mFlags;		//Flags used for calculating bounding box

		DArray<uint64> mMortonCodes;
	};
}
