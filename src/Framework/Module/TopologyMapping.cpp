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

	void TopologyMapping::updateImpl()
	{
		this->apply();
	}

}