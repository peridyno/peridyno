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

	bool TopologyMapping::updateImpl()
	{
		this->apply();

		return true;
	}

}