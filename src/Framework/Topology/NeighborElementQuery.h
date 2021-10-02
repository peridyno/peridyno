#pragma once
#include "Module/ComputeModule.h"
#include "Module/TopologyModule.h"
#include "SparseOctree.h"
#include "DiscreteElements.h"
#include "NeighborConstraints.h"

namespace dyno {
	template<typename TDataType> class CollisionDetectionBroadPhase;
	typedef typename TNeighborConstraints<Real> NeighborConstraints;


	class NbrFilter
	{
	public:
		bool sphere_sphere = true;
		bool sphere_box = true;
		bool sphere_tet = true;
		bool sphere_capsule = true;
		bool sphere_tri = true;
		bool box_box = true;
		bool box_tet = true;
		bool box_capsule = true;
		bool box_tri = true;
		bool tet_tet = true;
		bool tet_capsule = true;
		bool tet_tri = true;
		bool capsule_capsule = true;
		bool capsule_tri = true;
		bool tri_tri = true;
		bool tet_sdf = false;
		bool tet_neighbor_filter = true;
	};


	/**
	 * @brief A class implementation to find neighboring elements for a given array of elements
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
		DEF_VAR_IN(Real, Radius, "Search radius");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Particle position");
		

		/**
		 * @brief Ids of neighboring particles
		 */
		DEF_ARRAYLIST_OUT(int, Neighborhood, DeviceType::GPU, "Return neighbor ids");


		DeviceArrayField<NeighborConstraints> nbr_cons;
		NbrFilter Filter;

	private:
		DArray<AABB> m_queryAABB;
		DArray<AABB> m_queriedAABB;

		Scan m_scan;
		Reduction<int> m_reduce;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> m_broadPhaseCD;
		std::shared_ptr<DiscreteElements<TDataType>> discreteSet;
		std::shared_ptr<DiscreteElements<TDataType>> discreteSet_sorted;

		std::ofstream fout;
		
	};

#ifdef PRECISION_FLOAT
	template class NeighborElementQuery<DataType3f>;
#else
	template class NeighborElementQuery<DataType3d>;
#endif
}