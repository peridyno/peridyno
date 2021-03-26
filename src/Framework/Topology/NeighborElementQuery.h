#pragma once
#include "Framework/ModuleCompute.h"
#include "Topology/FieldNeighbor.h"
#include "Topology/GridHash.h"
#include "Framework/ModuleTopology.h"
#include "Topology/SparseOctree.h"
#include "DiscreteElements.h"
#include "NeighborConstraints.h"

namespace dyno {
	template<typename TDataType> class CollisionDetectionBroadPhase;
	typedef typename TNeighborConstraints<Real> NeighborConstraints;
	/**
	 * @brief A class implementation to find neighboring triangles for a given array of positions
	 * 
	 * @tparam TDataType 
	 */
	template<typename TDataType>
	class NeighborElementQuery : public ComputeModule
	{
		DECLARE_CLASS_1(NeighborElementQuery, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		NeighborElementQuery();
		~NeighborElementQuery() override;
		
		void compute() override;

		bool initializeImpl() override;
		void setDiscreteSet(std::shared_ptr<DiscreteElements<TDataType>> d)
		{
			discreteSet = d;
		}

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
		 * @brief Ids of neighboring particles
		 */
		DEF_EMPTY_OUT_NEIGHBOR_LIST(Neighborhood, int, "Neighboring particles' ids");


		DeviceArrayField<NeighborConstraints> nbr_cons;

	private:
		DArray<AABB> m_queryAABB;
		DArray<AABB> m_queriedAABB;

		Scan m_scan;
		Reduction<int> m_reduce;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> m_broadPhaseCD;
		std::shared_ptr<DiscreteElements<TDataType>> discreteSet;
	};

#ifdef PRECISION_FLOAT
	template class NeighborElementQuery<DataType3f>;
#else
	template class NeighborElementQuery<DataType3d>;
#endif
}