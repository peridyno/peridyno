#include "ModuleCompute.h"
#include "Framework/Node.h"

namespace dyno
{
ComputeModule::ComputeModule()
{
}

ComputeModule::~ComputeModule()
{
}

bool ComputeModule::execute()
{
	this->compute();

	return true;
}

}