#pragma once
#include "CollisionData.h"
#include "Attribute.h"

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
		typedef typename ::dyno::TOrientedBox3D<Real> Box3D;

		NeighborElementQuery();
		~NeighborElementQuery() override;

	public:
		DEF_VAR(bool, SelfCollision, true, "");

		DEF_VAR(Real, DHead, 0.0f, "D head");

		/**
		* @brief A positive value indicating the size of the smallest element, its value will also influence the level of Octree or hierarchical BVH
		*/
		DEF_VAR(Real, GridSizeLimit, Real(0.01),  "Indicate the size of the smallest element");

		DEF_INSTANCE_IN(DiscreteElements<TDataType>, DiscreteElements, "");

		DEF_ARRAY_IN(CollisionMask, CollisionMask, DeviceType::GPU, "");

		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "");

		DEF_ARRAY_OUT(TContactPair<Real>, Contacts, DeviceType::GPU, "");

	protected:
		void compute() override;

	private:
		DArray<AABB> mQueryAABB;
		DArray<AABB> mQueriedAABB;

		Scan<int> mScan;
		Reduction<int> mReduce;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> mBroadPhaseCD;
		std::shared_ptr<DiscreteElements<TDataType>> mDiscreteElements;		
	};
}