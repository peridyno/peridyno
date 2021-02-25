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

bool ConstraintModule::execute()
{
	return this->constrain();
}

}