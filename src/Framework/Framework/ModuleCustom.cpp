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

void CustomModule::updateImpl()
{
	this->applyCustomBehavior();
}

void CustomModule::applyCustomBehavior()
{

}

}