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

	void ComputeModule::updateImpl()
	{
		this->compute();
	}
}