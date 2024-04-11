/**
 * Copyright 2021-2023 Xiaowei He
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
#include "Module/ComputeModule.h"

#include "Topology/TriangleSet.h"

#include "Primitive/Primitive3D.h"

namespace dyno 
{
	template<typename TDataType> class CollisionDetectionBroadPhase;

	template<typename TDataType>
	class NeighborTriangleQuery : public ComputeModule
	{
		DECLARE_TCLASS(NeighborTriangleQuery, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename TAlignedBox3D<Real> AABB;

		NeighborTriangleQuery();
		~NeighborTriangleQuery() override;

	public:
		DECLARE_ENUM(Spatial,
			BVH = 0,
			OCTREE = 1);

		DEF_ENUM(Spatial, Spatial, Spatial::BVH, "");

		/**
		* @brief Search radius
		* A positive value representing the radius of neighborhood for each point
		*/
		DEF_VAR_IN(Real, Radius, "Search radius");

		/**
		 * @brief A set of points to be required from.
		 */
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "A set of points whose neighbors will be required for");

		/**
		 * @brief A set of points to be required from.
		 */
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "A set of triangles to be required from");

		/**
		 * @brief Ids of neighboring particles
		 */
		DEF_ARRAYLIST_OUT(int, NeighborIds, DeviceType::GPU, "Return neighbor ids");

	protected:
		void compute() override;

	private:
		DArray<AABB> mQueryAABB;
		DArray<AABB> mQueriedAABB;

		Reduction<uint> mReduce;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> mBroadPhaseCD;
	};
}