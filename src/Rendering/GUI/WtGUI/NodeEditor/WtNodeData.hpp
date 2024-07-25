#pragma once

#include <Wt/WLogger.h>
#include <Wt/WString.h>

#include "NodePort.h"

enum class PortShape
{
	Point,
	Bullet,
	Diamond,
	HollowDiamond
};

struct NodeDataType
{
	std::string id;
	std::string name;
	PortShape shape;
};

enum class PortType
{
	None,
	In,
	Out
};

enum class CntType
{
	Link,
	Break
};

static const int INVALID_PORT = -1;

using PortIndex = int;

struct Port
{
	PortType type;

	PortIndex index;

	Port() : type(PortType::None), index(INVALID_PORT) {}

	Port(PortType t, PortIndex i) : type(t), index(i) {}

	bool indexIsValid() { return index != INVALID_PORT; }

	bool portTypeIsValid() { return type != PortType::None; }
};

inline PortType oppositePort(PortType port)
{
	PortType result = PortType::None;

	switch (port)
	{
	case PortType::In:
		result = PortType::Out;
		break;

	case PortType::Out:
		result = PortType::In;
		break;

	default:
		break;
	}

	return result;
}

class WtNodeData
{
public:
	virtual ~WtNodeData() = default;

	virtual bool sameType(WtNodeData &nodeData) const
	{
		return (this->type().id == nodeData.type().id);
	}

	virtual NodeDataType type() const = 0;

	CntType connectionType()
	{
		return cType;
	}

	void setConnectionType(CntType t)
	{
		cType = t;
	}

private:
	CntType cType = CntType::Link;
};

using dyno::Node;
using dyno::NodePort;

class WtImportNode : public WtNodeData
{
public:
	WtImportNode() {}
	WtImportNode(NodePort *n)
		: node_port(n) {}

	NodeDataType type() const override
	{
		return NodeDataType{"nodeport",
							"NodePort",
							PortShape::Bullet};
	}

	NodePort *getNodePort() { return node_port; }

	bool isEmpty() { return node_port == nullptr; }

	bool sameType(WtNodeData &nodeData) const override;

private:
	NodePort *node_port = nullptr;
};

class WtExportNode : public WtNodeData
{
public:
	WtExportNode()
	{
	}

	WtExportNode(std::shared_ptr<Node> n)
		: export_node(n)
	{
	}

	NodeDataType type() const override
	{
		return NodeDataType{"nodeexport",
							"NodeExport",
							PortShape::Bullet};
	}

	inline std::shared_ptr<Node> getNode() { return export_node; }

	bool isEmpty() { return export_node == nullptr; }

	bool sameType(WtNodeData &nodeData) const override;

private:
	std::shared_ptr<Node> export_node = nullptr;
};