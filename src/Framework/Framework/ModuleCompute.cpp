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

	bool ComputeModule::updateImpl()
	{
		this->compute();

		return true;
	}
}