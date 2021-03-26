#pragma once
#include "Array/Array.h"
#include "Framework/CollisionModel.h"

namespace dyno
{
template <typename> class CollidablePoints;
template <typename> class NeighborQuery;
template <typename> class NeighborList;
template <typename> class GridHash;

template<typename TDataType>
class CollisionPoints : public CollisionModel
{
	DECLARE_CLASS_1(CollisionPoints, TDataType)
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	CollisionPoints();
	virtual ~CollisionPoints();

	bool isSupport(std::shared_ptr<CollidableObject> obj) override;

	void addCollidableObject(std::shared_ptr<CollidableObject> obj) override;

	bool initializeImpl() override;

	void doCollision() override;
	
protected:
	DArray<int> m_objId;
	DArray<Coord> m_points;
	DArray<Coord> m_vels;

	std::shared_ptr<NeighborQuery<TDataType>> m_nbrQuery;
	std::shared_ptr<NeighborList<int>> m_nList;

	std::vector<std::shared_ptr<CollidablePoints<TDataType>>> m_collidableObjects;
};

#ifdef PRECISION_FLOAT
template class CollisionPoints<DataType3f>;
#else
template class CollisionPoints<DataType3d>;
#endif

}
