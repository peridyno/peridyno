#include "ModuleConstraint.h"
#include "Framework/Node.h"

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