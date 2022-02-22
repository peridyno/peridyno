#include "NodePort.h"
#include "Node.h"

namespace dyno
{

	NodePort::NodePort(std::string name, std::string description, Node* parent /*= nullptr*/)
		: m_name(name)
		, m_description(description)
		, m_portType(NodePortType::Unknown)
		, m_parent(parent)
	{
		parent->addNodePort(this);
	}

	NodePortType NodePort::getPortType()
	{
		return m_portType;
	}

	void NodePort::setPortType(NodePortType portType)
	{
		m_portType = portType;
	}

	void NodePort::clear()
	{
		m_nodes.clear();
	}
}

