#include "CollisionModel.h"
#include "Node.h"

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