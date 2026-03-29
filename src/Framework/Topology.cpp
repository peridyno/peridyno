#include "Topology.h"

namespace dyno
{
Topology::Topology()
	: Object()
	, m_topologyChanged(true)
{

}

Topology::~Topology()
{
}

void Topology::update()
{
	this->updateTopology();
}

}