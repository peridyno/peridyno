#include "Module/TopologyModule.h"

namespace dyno
{
TopologyModule::TopologyModule()
	: Object()
	, m_topologyChanged(true)
{

}

TopologyModule::~TopologyModule()
{
}

void TopologyModule::update()
{
	this->updateTopology();
}

}