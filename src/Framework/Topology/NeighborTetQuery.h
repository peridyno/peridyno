#pragma once
#include "Framework/ModuleCompute.h"
#include "Framework/FieldVar.h"
#include "Framework/FieldArray.h"
#include "Topology/FieldNeighbor.h"
#include "Topology/GridHash.h"
#include "Framework/ModuleTopology.h"
#include "Topology/SparseOctree.h"
#include "DiscreteElements.h"
#include "NeighborConstraints.h"
#include "TriangleSet.h"
#include "TetrahedronSet.h"

namespace dyno {
	template<typename TDataType> class CollisionDetectionBroadPhase;
	typedef typename TNeighborConstraints<Real> NeighborConstraints;
	/**
	 * @brief A class implementation to find neighboring triangles for a given array of positions
	 * 
	 * @tparam TDataType 
	 */
	template<typename TDataType>
	class NeighborTetQuery : public ComputeModule
	{
		DECLARE_CLASS_1(NeighborTetQuery, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		NeighborTetQuery();
		~NeighborTetQuery() override;
		
		void compute() override;

		bool initializeImpl() override;
		void setTetSet(std::shared_ptr<TetrahedronSet<TDataType>> d)
		{
			tetSet = d;
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
		GArray<AABB> m_queryAABB;
		GArray<AABB> m_queriedAABB;

		Scan m_scan;
		Reduction<int> m_reduce;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> m_broadPhaseCD;
		std::shared_ptr<TetrahedronSet<TDataType>> tetSet;
	};

#ifdef PRECISION_FLOAT
	template class NeighborTetQuery<DataType3f>;
#else
	template class NeighborTetQuery<DataType3d>;
#endif
}