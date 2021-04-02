#pragma once
#include "Framework/ModuleCompute.h"
#include "Topology/FieldNeighbor.h"
#include "Topology/GridHash.h"
#include "Framework/ModuleTopology.h"
#include "Topology/SparseOctree.h"


namespace dyno {
	template<typename TDataType> class CollisionDetectionBroadPhase;
	typedef typename TopologyModule::Triangle TriangleIndex;


	/**
	 * @brief A class implementation to find neighboring triangles for a given array of positions
	 * 
	 * @tparam TDataType 
	 */
	template<typename TDataType>
	class NeighborTriangleQuery : public ComputeModule
	{
		DECLARE_CLASS_1(NeighborTriangleQuery, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		NeighborTriangleQuery();
		~NeighborTriangleQuery() override;
		
		void compute() override;

		bool initializeImpl() override;

	public:
		/**
		* @brief Search radius
		* A positive value representing the radius of neighborhood for each point
		*/
		DEF_EMPTY_IN_VAR(Radius, Real, "Search radius");

		/**
		 * @brief Particle position
		 */
		DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");
		
		/**
		* @brief Triangle position
		*/
		DEF_EMPTY_IN_ARRAY(TriangleVertex, Coord, DeviceType::GPU, "Particle position");
		/**
		* @brief Triangle index
		*/
		DEF_EMPTY_IN_ARRAY(TriangleIndex, TriangleIndex, DeviceType::GPU, "Particle position");

		/**
		 * @brief Ids of neighboring particles
		 */
		DEF_EMPTY_OUT_NEIGHBOR_LIST(Neighborhood, int, "Neighboring particles' ids");


	private:
		DArray<AABB> m_queryAABB;
		DArray<AABB> m_queriedAABB;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> m_broadPhaseCD;
	};
}