#include "Framework/ModuleTopology.h"
#include "Framework/Node.h"

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

}