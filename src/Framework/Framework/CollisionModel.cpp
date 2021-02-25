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

bool CollisionModel::execute()
{
	this->doCollision();

	return true;
}

}