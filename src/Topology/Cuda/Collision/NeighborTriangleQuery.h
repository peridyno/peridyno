#pragma once
#include "Module/ComputeModule.h"
#include "Module/TopologyModule.h"
#include "Topology/SparseOctree.h"

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

		NeighborTriangleQuery();
		~NeighborTriangleQuery() override;
		
		void compute() override;

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
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "A set of points to be required from");

		
		/**
		 * @brief A set of points to be required from.
		 */
		DEF_ARRAY_IN(Coord, TriPosition, DeviceType::GPU, "A set of Triangles to be required from");

		/**
		 * @brief A set of points to be required from.
		 */
		DEF_ARRAY_IN(Triangle, Triangles, DeviceType::GPU, "A set of Triangles to be required from");


		/**
		 * @brief Ids of neighboring particles
		 */
		DEF_ARRAYLIST_OUT(int, NeighborIds, DeviceType::GPU, "Return neighbor ids");

	private:
		DArray<AABB> mQueryAABB;
		DArray<AABB> mQueriedAABB;

		Reduction<uint> mReduce;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> mBroadPhaseCD;
	};
}