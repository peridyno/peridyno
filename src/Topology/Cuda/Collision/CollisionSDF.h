#pragma once
#include "Module/CollisionModel.h"

namespace dyno
{
template <typename> class CollidableSDF;

template<typename TDataType>
class CollisionSDF : public CollisionModel
{
	DECLARE_TCLASS(CollisionSDF, TDataType)
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	CollisionSDF();
	virtual ~CollisionSDF();

	bool isSupport(std::shared_ptr<CollidableObject> obj) override;
	void addDrivenObject(std::shared_ptr<CollidableObject> obj);
	void setCollidableSDF(std::shared_ptr<CollidableObject> sdf) { m_cSDF = std::dynamic_pointer_cast<CollidableSDF<TDataType>>(sdf); }

	void addCollidableObject(std::shared_ptr<CollidableObject> obj) override;

	bool initializeImpl() override;

	void doCollision() override;
	
protected:
	Real m_normal_friction;
	Real m_tangent_friction;

	std::shared_ptr<CollidableSDF<TDataType>> m_cSDF;
	std::vector<std::shared_ptr<CollidableObject>> m_collidableObjects;
};
}
