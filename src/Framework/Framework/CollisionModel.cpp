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

bool CollisionModel::updateImpl()
{
	this->doCollision();

	return true;
}

}