#pragma once
#include "CollisionData.h"

#include "Module/ComputeModule.h"

#include "Topology/DiscreteElements.h"

namespace dyno {
	template<typename TDataType> class CollisionDetectionBroadPhase;
	/**
	 * @brief A class implementation to find neighboring elements for a given array of elements
	 * 
	 * @tparam TDataType 
	 */
	template<typename TDataType>
	class NeighborElementQuery : public ComputeModule
	{
		DECLARE_TCLASS(NeighborElementQuery, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename ::dyno::TAlignedBox3D<Real> AABB;

		NeighborElementQuery();
		~NeighborElementQuery() override;
		
		void compute() override;

	public:
		/**
		* @brief Search radius
		* A positive value representing the radius of neighborhood for each point
		*/
		DEF_VAR_IN(Real, Radius, "Search radius");

		DEF_INSTANCE_IN(DiscreteElements<TDataType>, DiscreteElements, "");

		DEF_ARRAY_IN(CollisionMask, CollisionMask, DeviceType::GPU, "");

		DEF_ARRAY_OUT(TContactPair<Real>, Contacts, DeviceType::GPU, "");
	private:
		DArray<AABB> mQueryAABB;
		DArray<AABB> mQueriedAABB;

		Scan<int> mScan;
		Reduction<int> mReduce;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> mBroadPhaseCD;
		std::shared_ptr<DiscreteElements<TDataType>> mDiscreteElements;		
		int cnt = 0;
	};
}