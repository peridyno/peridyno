#include "Module/TopologyModule.h"
#include "Node.h"

namespace dyno
{
IMPLEMENT_CLASS(TopologyModule)

TopologyModule::TopologyModule()
	: Module()
	, m_topologyChanged(true)
{

}

TopologyModule::~TopologyModule()
{
}

void TopologyModule::updateImpl()
{
	this->updateTopology();
}

}