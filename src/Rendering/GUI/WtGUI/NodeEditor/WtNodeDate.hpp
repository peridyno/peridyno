#pragma once

#include <Wt/WLogger.h>
#include <Wt/WString.h>

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
	case PortType::In:result = PortType::Out;
		break;

	case PortType::Out:result = PortType::In;
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

	virtual bool sameType(WtNodeData& nodeData) const
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