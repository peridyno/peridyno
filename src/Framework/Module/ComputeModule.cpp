#include "ComputeModule.h"
#include "Node.h"

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