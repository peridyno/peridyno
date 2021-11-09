#pragma once
#include "Module/ComputeModule.h"
#include "Module/TopologyModule.h"
#include "Topology/SparseOctree.h"

namespace dyno 
{
	template<typename TDataType> class CollisionDetectionBroadPhase;

	template<typename TDataType>
	class NeighborTriQueryOctree : public ComputeModule
	{
		DECLARE_CLASS_1(NeighborTriQueryOctree, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		NeighborTriQueryOctree();
		~NeighborTriQueryOctree() override;
		
		void compute() override;

	public:
		//DEF_VAR(SizeLimit, uint, 0, "Maximum number of neighbors");

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
		DArray<AABB> m_queryAABB;
		DArray<AABB> m_queriedAABB;

		Reduction<int> m_reduce;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> m_broadPhaseCD;
	};
}