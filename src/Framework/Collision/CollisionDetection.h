#pragma once
#include "Framework/CollisionModel.h"
#include "Topology/Primitive3D.h"

namespace dyno
{
template <typename> class CollisionDetectionBroadPhase;

typedef typename TAlignedBox3D<Real> AABB;

template<typename TDataType>
class CollisionDetection : public CollisionModel
{
	DECLARE_CLASS_1(CollisionDetection, TDataType)
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename TDataType::Matrix Matrix;

	CollisionDetection();
	virtual ~CollisionDetection();

	void doCollision() override;

	bool initializeImpl() override;

	bool isSupport(std::shared_ptr<CollidableObject> obj);

public:
	
	DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "");

	DEF_EMPTY_IN_ARRAY(Rotation, Matrix, DeviceType::GPU, "");

	DEF_EMPTY_OUT_ARRAY(ContactPairs, ContactPair, DeviceType::GPU, "Contact pairs");

private:
	std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> m_broadPhaseCD;
};
}
