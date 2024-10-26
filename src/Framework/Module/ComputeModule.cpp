#include "ComputeModule.h"

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