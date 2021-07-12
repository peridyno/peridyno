#include "ConstraintModule.h"
#include "Node.h"

namespace dyno
{
ConstraintModule::ConstraintModule()
	: Module()
{
}

ConstraintModule::~ConstraintModule()
{
}

void ConstraintModule::updateImpl()
{
	this->constrain();
}

}