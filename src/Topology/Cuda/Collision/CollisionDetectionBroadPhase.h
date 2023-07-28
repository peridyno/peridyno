#pragma once
#include "Module/ComputeModule.h"

#include "Algorithm/Reduction.h"
#include "Primitive/Primitive3D.h"


namespace dyno
{
	typedef typename ::dyno::TAlignedBox3D<Real> AABB;
	typedef unsigned long long int PKey;

	template<typename TDataType>
	class CollisionDetectionBroadPhase : public ComputeModule
	{
		DECLARE_TCLASS(CollisionDetectionBroadPhase, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		CollisionDetectionBroadPhase();
		virtual ~CollisionDetectionBroadPhase();

		void setSelfCollision(bool s)
		{
			self_collision = s;
		}

	public:
		DECLARE_ENUM(EStructure,
			BVH = 0,
			Octree = 1);

		DEF_ENUM(EStructure, AccelerationStructure, EStructure::BVH, "Acceleration structure");

		DEF_VAR(Real, GridSizeLimit, 0.005, "Limit the smallest grid size");

		DEF_ARRAY_IN(AABB, Source, DeviceType::GPU, "");

		DEF_ARRAY_IN(AABB, Target, DeviceType::GPU, "");

		DEF_ARRAYLIST_OUT(int, ContactList, DeviceType::GPU, "Contact pairs");

	protected:
		void compute() override;

	private:
		void doCollisionWithSparseOctree();
		void doCollisionWithLinearBVH();

	private:
		Reduction<Real> m_reduce_real;
		Reduction<Coord> m_reduce_coord;

		bool self_collision = false;

		DArray<Real> mH;

		DArray<Coord> mV0;
		DArray<Coord> mV1;

		DArray<uint> mCounter;
		DArray<uint> mNewCounter;

		DArray<int> mIds;
		DArray<PKey> mKeys;
	};

	IMPLEMENT_TCLASS(CollisionDetectionBroadPhase, TDataType)
}
