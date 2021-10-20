#include "Module/OutputModule.h"

namespace dyno
{
	OutputModule::OutputModule()
		: Module()
	{

	}

	OutputModule::~OutputModule()
	{
	}

	void OutputModule::updateImpl()
	{
		this->flush();
	}
}