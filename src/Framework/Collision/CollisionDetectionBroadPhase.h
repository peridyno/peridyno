#pragma once
#include "Framework/CollisionModel.h"
#include "Topology/Primitive3D.h"
#include "Topology/SparseOctree.h"
#include "Algorithm/Reduction.h"

namespace dyno
{
	typedef typename TAlignedBox3D<Real> AABB;

	template<typename TDataType>
	class CollisionDetectionBroadPhase : public CollisionModel
	{
		DECLARE_CLASS_1(CollisionDetectionBroadPhase, TDataType)
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

		DEF_EMPTY_VAR(GridSizeLimit, Real, "Limit the smallest grid size");

		DEF_EMPTY_IN_ARRAY(Source, AABB, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(Target, AABB, DeviceType::GPU, "");


		DEF_EMPTY_OUT_NEIGHBOR_LIST(ContactList, int, "Contact pairs");


	private:
		Reduction<Real> m_reduce_real;
		Reduction<Coord> m_reduce_coord;

		SparseOctree<TDataType> octree;
		bool self_collision = false;
	};

#ifdef PRECISION_FLOAT
	template class CollisionDetectionBroadPhase<DataType3f>;
#else
	template class CollisionDetectionBroadPhase<DataType3d>;
#endif

}
