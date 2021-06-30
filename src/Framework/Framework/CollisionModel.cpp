#include "CollisionModel.h"
#include "Framework/Node.h"

namespace dyno
{

CollisionModel::CollisionModel()
{
}

CollisionModel::~CollisionModel()
{
}

void CollisionModel::updateImpl()
{
	this->doCollision();
}

}