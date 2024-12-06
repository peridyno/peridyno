#pragma once

#include <Wt/WPointF.h>
#include <Wt/WRectF.h>

#include "guid.hpp"

#include "WtNodeData.hpp"

//#include "QtSerializable.hpp"
#include "WtDataModelRegistry.h"
#include "Export.hpp"

class WtNode;
class WtNodeData;
class WtConnectionGraphicsObject;
class WtNode;
class WtConnectionGeometry;

class WtConnectionGeometry
{
public:

	WtConnectionGeometry();

public:

	Wt::WPointF const& getEndPoint(PortType portType) const;

	void setEndPoint(PortType portType, Wt::WPointF const& point);

	void moveEndPoint(PortType portType, Wt::WPointF const& offset);

	Wt::WRectF boundingRect() const;

	std::pair<Wt::WPointF, Wt::WPointF> pointsC1C2() const;

	Wt::WPointF source() const { return _out; }
	Wt::WPointF sink() const { return _in; }

	double lineWidth() const { return _lineWidth; }

	bool hovered() const { return _hovered; }
	void setHovered(bool hovered) { _hovered = hovered; }

	Wt::WPointF out()
	{
		return _out;
	}

	Wt::WPointF in()
	{
		return _in;
	}

private:
	// local object coordinates
	Wt::WPointF _in;
	Wt::WPointF _out;

	//int _animationPhase;

	double _lineWidth;

	bool _hovered;
};


/// Stores currently draggind end.
/// Remembers last hovered WtNode
class WtConnectionState
{
public:

	WtConnectionState(PortType port = PortType::None)
		: _requiredPort(port) {}

	WtConnectionState(const WtConnectionState&) = delete;
	WtConnectionState operator=(const WtConnectionState&) = delete;

	~WtConnectionState();

public:

	void setRequiredPort(PortType end)
	{
		_requiredPort = end;
	}

	PortType requiredPort() const
	{
		return _requiredPort;
	}

	bool requiresPort() const
	{
		return _requiredPort != PortType::None;
	}

	void setNoRequiredPort()
	{
		_requiredPort = PortType::None;
	}

public:

	void interactWithNode(WtNode* node);

	void setLastHoveredNode(WtNode* node);

	WtNode* lastHoveredNode() const
	{
		return _lastHoveredNode;
	}

	void resetLastHoveredNode();

private:

	PortType _requiredPort;

	WtNode* _lastHoveredNode{ nullptr };
};

class WtConnection
{
public:

	/// New WtConnection is attached to the port of the given WtNode.
	/// The port has parameters (portType, portIndex).
	/// The opposite connection end will require anothre port.
	WtConnection(
		PortType portType,
		WtNode& node,
		PortIndex portIndex);

	WtConnection(
		WtNode& nodeIn,
		PortIndex portIndexIn,
		WtNode& nodeOut,
		PortIndex portIndexOut,
		TypeConverter converter =
		TypeConverter{});

	WtConnection(
		WtNode& nodeIn,
		PortIndex portIndexIn,
		WtNode& nodeOut,
		PortIndex portIndexOut);

	WtConnection(const WtConnection&) = delete;
	WtConnection operator=(const WtConnection&) = delete;

	~WtConnection();

	//public:
	//
	//	QJsonObject
	//		save() const override;

public:

	Wt::Guid id() const;

	/// Remembers the end being dragged.
	/// Invalidates WtNode address.
	/// Grabs mouse.
	void setRequiredPort(PortType portType);

	PortType requiredPort() const;

	void setGraphicsObject(std::unique_ptr<WtConnectionGraphicsObject>&& graphics);

	/// Assigns a node to the required port.
	/// It is assumed that there is a required port, no extra checks
	void setNodeToPort(WtNode& node,
		PortType portType,
		PortIndex portIndex);

	void removeFromNodes() const;

public:

	WtConnectionGraphicsObject& getConnectionGraphicsObject() const;

	WtConnectionState const& connectionState() const;

	WtConnectionState& connectionState();

	WtConnectionGeometry& connectionGeometry();

	WtConnectionGeometry const& connectionGeometry() const;

	WtNode* getNode(PortType portType) const;

	WtNode*& getNode(PortType portType);

	PortIndex getPortIndex(PortType portType) const;

	void clearNode(PortType portType);

	NodeDataType dataType(PortType portType) const;

	void setTypeConverter(TypeConverter converter);

	bool complete() const;

public: // data propagation

	void propagateData(std::shared_ptr<WtNodeData> nodeData) const;

	void propagateEmptyData() const;

	void propagateDisconnectedData() const;

	//Q_SIGNALS:
	//
	//	void
	//		connectionCompleted(WtConnection const&) const;
	//
	//	void
	//		connectionMadeIncomplete(WtConnection const&) const;

private:

	Wt::Guid _uid;

private:

	WtNode* _outNode = nullptr;
	WtNode* _inNode = nullptr;

	PortIndex _outPortIndex;
	PortIndex _inPortIndex;

private:

	WtConnectionState    _connectionState;
	WtConnectionGeometry _connectionGeometry;

	std::unique_ptr<WtConnectionGraphicsObject>_connectionGraphicsObject;

	TypeConverter _converter;

	//Q_SIGNALS:
	//
	//	void
	//		updated(WtConnection& conn) const;
};