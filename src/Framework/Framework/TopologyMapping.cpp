#pragma once
#include "TopologyMapping.h"

namespace dyno
{
	TopologyMapping::TopologyMapping()
		: Module()
	{

	}

	TopologyMapping::~TopologyMapping()
	{

	}

	bool TopologyMapping::execute()
	{
		this->apply();

		return true;
	}

}