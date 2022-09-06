#pragma once
#include "Module/CollisionModel.h"
#include "Algorithm/Reduction.h"
#include "Topology/Primitive3D.h"
#include "Topology/SparseOctree.h"

namespace dyno
{
	typedef typename TAlignedBox3D<Real> AABB;

	typedef unsigned long long int PKey;

	template<typename TDataType>
	class CollisionDetectionBroadPhase : public CollisionModel
	{
		DECLARE_TCLASS(CollisionDetectionBroadPhase, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		CollisionDetectionBroadPhase();
		virtual ~CollisionDetectionBroadPhase();

		void doCollision() override;

		bool isSupport(std::shared_ptr<CollidableObject> obj) override { return true; }
		void setSelfCollision(bool s)
		{
			self_collision = s;
		}

	public:
		DEF_VAR(Real, GridSizeLimit, 0.005, "Limit the smallest grid size");

		DEF_ARRAY_IN(AABB, Source, DeviceType::GPU, "");

		DEF_ARRAY_IN(AABB, Target, DeviceType::GPU, "");

		DEF_ARRAYLIST_OUT(int, ContactList, DeviceType::GPU, "Contact pairs");


	private:
		Reduction<Real> m_reduce_real;
		Reduction<Coord> m_reduce_coord;

		SparseOctree<TDataType> octree;
		bool self_collision = false;

		DArray<Real> mH;

		DArray<Coord> mV0;
		DArray<Coord> mV1;

		DArray<int> mCounter;
		DArray<int> mNewCounter;

		DArray<int> mIds;
		DArray<PKey> mKeys;
	};

	IMPLEMENT_TCLASS(CollisionDetectionBroadPhase, TDataType)
}
