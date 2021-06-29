#include "ModuleCustom.h"
#include "Framework/Node.h"

namespace dyno
{
IMPLEMENT_CLASS(CustomModule)

CustomModule::CustomModule()
	: Module()
{
}

CustomModule::~CustomModule()
{
}

bool CustomModule::updateImpl()
{
	this->applyCustomBehavior();
	return true;
}

void CustomModule::applyCustomBehavior()
{

}

}