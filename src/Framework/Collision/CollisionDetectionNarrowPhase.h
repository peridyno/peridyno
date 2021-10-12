#pragma once
#include "Module/CollisionModel.h"
#include "Topology/Primitive3D.h"

#include "CollisionData.h"

namespace dyno
{
template <typename> class CollisionDetectionBroadPhase;

typedef typename TAlignedBox3D<Real> AABB;

template<typename TDataType>
class CollisionDetectionNarrowPhase : public CollisionModel
{
	DECLARE_CLASS_1(CollisionDetectionNarrowPhase, TDataType)
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename TDataType::Matrix Matrix;
	typedef typename TContactPair<Real> ContactPair;

	CollisionDetectionNarrowPhase();
	virtual ~CollisionDetectionNarrowPhase();

	void doCollision() override;

	bool isSupport(std::shared_ptr<CollidableObject> obj);

public:
	
	DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

	DEF_ARRAY_IN(Matrix, Rotation, DeviceType::GPU, "");

	DEF_ARRAY_OUT(ContactPair, ContactPairs, DeviceType::GPU, "Contact pairs");

private:
	std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> m_broadPhaseCD;
};
}
