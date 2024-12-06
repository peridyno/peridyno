#include "WtConnection.h"

#include <cmath>
#include <utility>
#include <cassert>

#include "WtNode.h"
#include "WtFlowScene.h"

#include "WtNodeGraphicsObject.h"
#include "WtNodeDataModel.h"
#include "WtConnectionGraphicsObject.h"

#include "WtNodeStyle.h"

WtConnectionGeometry::WtConnectionGeometry()
	: _in(0, 0)
	, _out(0, 0)
	, _lineWidth(3.0)
	, _hovered(false)
{}

Wt::WPointF const& WtConnectionGeometry::getEndPoint(PortType portType) const
{
	assert(portType != PortType::None);
	return (portType == PortType::Out ? _out : _in);
}

void WtConnectionGeometry::setEndPoint(PortType portType, Wt::WPointF const& point)
{
	switch (portType)
	{
	case PortType::In:
		_in = point;
		break;
	case PortType::Out:
		_out = point;
		break;

	default:
		break;
	}
}

void WtConnectionGeometry::moveEndPoint(PortType portType, Wt::WPointF const& offset)
{
	switch (portType)
	{
	case PortType::In:
		_in += offset;
		break;
	case PortType::Out:
		_out += offset;
		break;

	default:
		break;
	}
}

Wt::WRectF WtConnectionGeometry::boundingRect() const
{
	auto points = pointsC1C2();

	Wt::WRectF basicRect = Wt::WRectF(_out, _in).normalized();

	Wt::WRectF c1c2Rect = Wt::WRectF(points.first, points.second).normalized();

	auto const& connectionStyle = WtStyleCollection::connectionStyle();

	float const diam = connectionStyle.pointDiameter();

	Wt::WRectF commonRect = basicRect.united(c1c2Rect);

	Wt::WPointF const cornerOffset(diam, diam);

	Wt::WPointF topLeft(commonRect.topLeft().x() - cornerOffset.x(), commonRect.topLeft().y() - cornerOffset.y());

	Wt::WPointF bottomRight(commonRect.bottomLeft().x() + 2 * cornerOffset.x(), commonRect.bottomLeft().y() + 2 * cornerOffset.y());

	return Wt::WRectF(topLeft, bottomRight);
}

std::pair<Wt::WPointF, Wt::WPointF> WtConnectionGeometry::pointsC1C2() const
{
	const double defaultOffset = 200;

	double xDistance = _in.x() - _out.x();

	double horizontalOffset = std::min(defaultOffset, std::abs(xDistance));

	double verticalOffset = 0;

	double ratioX = 0.5;

	if (xDistance <= 0)
	{
		double yDistance = _in.y() - _out.y();

		double vector = yDistance < 0 ? -1.0 : 1.0;

		verticalOffset = std::min(defaultOffset, std::abs(yDistance)) * vector;

		ratioX = 1.0;
	}

	horizontalOffset *= ratioX;

	Wt::WPointF c1(_out.x() + horizontalOffset, _out.y() + verticalOffset);
	Wt::WPointF c2(_in.x() - horizontalOffset, _in.y() - verticalOffset);

	return std::make_pair(c1, c2);
}

// WtConnectionState
WtConnectionState::~WtConnectionState()
{
	resetLastHoveredNode();
}

void WtConnectionState::interactWithNode(WtNode* node)
{
	if (node)
	{
		_lastHoveredNode = node;
	}
	else
	{
		resetLastHoveredNode();
	}
}

void WtConnectionState::setLastHoveredNode(WtNode* node)
{
	_lastHoveredNode = node;
}

void WtConnectionState::resetLastHoveredNode()
{
	if (_lastHoveredNode)
		_lastHoveredNode->resetReactionToConnection();

	_lastHoveredNode = nullptr;
}

WtConnection::WtConnection(
	PortType portType,
	WtNode& node,
	PortIndex portIndex)
	: _uid(Wt::newGuid())
	, _outPortIndex(INVALID_PORT)
	, _inPortIndex(INVALID_PORT)
	, _connectionState()
{
	setNodeToPort(node, portType, portIndex);

	setRequiredPort(oppositePort(portType));
}

WtConnection::WtConnection(
	WtNode& nodeIn,
	PortIndex portIndexIn,
	WtNode& nodeOut,
	PortIndex portIndexOut,
	TypeConverter converter)
	: _uid(Wt::Guid())
	, _outNode(&nodeOut)
	, _inNode(&nodeIn)
	, _outPortIndex(portIndexOut)
	, _inPortIndex(portIndexIn)
	, _connectionState()
{
	setNodeToPort(nodeIn, PortType::In, portIndexIn);
	setNodeToPort(nodeOut, PortType::Out, portIndexOut);
}

WtConnection::WtConnection(
	WtNode& nodeIn,
	PortIndex portIndexIn,
	WtNode& nodeOut,
	PortIndex portIndexOut)
	: _uid(Wt::Guid())
	, _outNode(&nodeOut)
	, _inNode(&nodeIn)
	, _outPortIndex(portIndexOut)
	, _inPortIndex(portIndexIn)
	, _connectionState()
{
	setNodeToPort(nodeIn, PortType::In, portIndexIn);
	setNodeToPort(nodeOut, PortType::Out, portIndexOut);
}

WtConnection::~WtConnection()
{
	if (complete())
	{
		//signal
		//connectionMadeIncomplete(*this);
	}
	//propagateDisconnectedData();

	if (_inNode)
	{
		// No Update
		//_inNode->nodeGraphicsObject().update();
	}

	if (_outNode)
	{
		// No Update
		//_outNode->nodeGraphicsObject.update();
	}
}

//QJsonObject
//QtConnection::
//save() const
//{
//	QJsonObject connectionJson;
//
//	if (_inNode && _outNode)
//	{
//		connectionJson["in_id"] = _inNode->id().toString();
//		connectionJson["in_index"] = _inPortIndex;
//
//		connectionJson["out_id"] = _outNode->id().toString();
//		connectionJson["out_index"] = _outPortIndex;
//
//		if (_converter)
//		{
//			auto getTypeJson = [this](PortType type)
//				{
//					QJsonObject typeJson;
//					NodeDataType nodeType = this->dataType(type);
//					typeJson["id"] = nodeType.id;
//					typeJson["name"] = nodeType.name;
//
//					return typeJson;
//				};
//
//			QJsonObject converterTypeJson;
//
//			converterTypeJson["in"] = getTypeJson(PortType::In);
//			converterTypeJson["out"] = getTypeJson(PortType::Out);
//
//			connectionJson["converter"] = converterTypeJson;
//		}
//	}
//
//	return connectionJson;
//}

Wt::Guid WtConnection::id() const
{
	return _uid;
}

bool WtConnection::complete() const
{
	return _inNode != nullptr && _outNode != nullptr;
}

void WtConnection::setRequiredPort(PortType dragging)
{
	_connectionState.setRequiredPort(dragging);

	switch (dragging)
	{
	case PortType::Out:
		_outNode = nullptr;
		_outPortIndex = INVALID_PORT;
		break;

	case PortType::In:
		_inNode = nullptr;
		_inPortIndex = INVALID_PORT;
		break;

	default:
		break;
	}
}

PortType WtConnection::requiredPort() const
{
	return _connectionState.requiredPort();
}

void WtConnection::setGraphicsObject(std::unique_ptr<WtConnectionGraphicsObject>&& graphics)
{
	_connectionGraphicsObject = std::move(graphics);

	// This function is only called when the ConnectionGraphicsObject
	// is newly created. At this moment both end coordinates are (0, 0)
	// in QtConnection G.O. coordinates. The position of the whole
	// QtConnection G. O. in scene coordinate system is also (0, 0).
	// By moving the whole object to the QtNode Port position
	// we position both connection ends correctly.

	if (requiredPort() != PortType::None)
	{
		PortType attachedPort = oppositePort(requiredPort());

		PortIndex attachedPortIndex = getPortIndex(attachedPort);

		auto node = getNode(attachedPort);

		Wt::WTransform nodeSceneTransform = node->nodeGraphicsObject().sceneTransform();

		Wt::WPointF pos = node->nodeGeometry().portScenePosition(attachedPortIndex,
			attachedPort,
			nodeSceneTransform);

		_connectionGraphicsObject->setPos(pos);
	}
	_connectionGraphicsObject->move();
}

PortIndex WtConnection::getPortIndex(PortType portType) const
{
	PortIndex result = INVALID_PORT;

	switch (portType)
	{
	case PortType::In:
		result = _inPortIndex;
		break;

	case PortType::Out:
		result = _outPortIndex;

		break;

	default:
		break;
	}

	return result;
}

void WtConnection::setNodeToPort(
	WtNode& node,
	PortType portType,
	PortIndex portIndex)
{
	bool wasIncomplete = !complete();

	auto& nodeWeak = getNode(portType);

	nodeWeak = &node;

	if (portType == PortType::Out)
		_outPortIndex = portIndex;
	else
		_inPortIndex = portIndex;

	_connectionState.setNoRequiredPort();

	//signal
	//updated(*this);

	if (complete() && wasIncomplete) {
		//signal
		//connectionCompleted(*this);
	}
}

void WtConnection::removeFromNodes() const
{
	if (_inNode)
		_inNode->nodeState().eraseConnection(PortType::In, _inPortIndex, id());

	if (_outNode)
		_outNode->nodeState().eraseConnection(PortType::Out, _outPortIndex, id());
}

WtConnectionGraphicsObject& WtConnection::getConnectionGraphicsObject() const
{
	return *_connectionGraphicsObject;
}

WtConnectionState& WtConnection::connectionState()
{
	return _connectionState;
}

WtConnectionState const& WtConnection::connectionState() const
{
	return _connectionState;
}

WtConnectionGeometry& WtConnection::connectionGeometry()
{
	return _connectionGeometry;
}

WtConnectionGeometry const& WtConnection::connectionGeometry() const
{
	return _connectionGeometry;
}

WtNode* WtConnection::getNode(PortType portType) const
{
	switch (portType)
	{
	case PortType::In:
		return _inNode;
		break;

	case PortType::Out:
		return _outNode;
		break;

	default:
		// not possible
		break;
	}
	return nullptr;
}

WtNode*& WtConnection::getNode(PortType portType)
{
	switch (portType)
	{
	case PortType::In:
		return _inNode;
		break;

	case PortType::Out:
		return _outNode;
		break;

	default:
		// not possible
		break;
	}
	//sQ_UNREACHABLE();
}

void WtConnection::clearNode(PortType portType)
{
	if (complete())
	{
		//signal
		//connectionMadeIncomplete(*this);
	}

	getNode(portType) = nullptr;

	if (portType == PortType::In)
		_inPortIndex = INVALID_PORT;
	else
		_outPortIndex = INVALID_PORT;
}

NodeDataType WtConnection::dataType(PortType portType) const
{
	if (_inNode && _outNode)
	{
		auto const& model = (portType == PortType::In) ?
			_inNode->nodeDataModel() :
			_outNode->nodeDataModel();
		PortIndex index = (portType == PortType::In) ?
			_inPortIndex :
			_outPortIndex;

		return model->dataType(portType, index);
	}
	else
	{
		WtNode* validNode;
		PortIndex index = INVALID_PORT;

		if ((validNode = _inNode))
		{
			index = _inPortIndex;
			portType = PortType::In;
		}
		else if ((validNode = _outNode))
		{
			index = _outPortIndex;
			portType = PortType::Out;
		}

		if (validNode)
		{
			auto const& model = validNode->nodeDataModel();

			return model->dataType(portType, index);
		}
	}
	//Q_UNREACHABLE();
}

void WtConnection::setTypeConverter(TypeConverter converter)
{
	_converter = std::move(converter);
}

void WtConnection::propagateData(std::shared_ptr<WtNodeData> nodeData) const
{
	if (_inNode)
	{
		if (_converter)
		{
			nodeData = _converter(nodeData);
		}

		_inNode->propagateData(nodeData, _inPortIndex);
	}
}

void WtConnection::propagateEmptyData() const
{
	std::shared_ptr<WtNodeData> emptyData;

	propagateData(emptyData);
}

void WtConnection::propagateDisconnectedData() const
{
	std::shared_ptr<WtNodeData> deletedData = nullptr;
	if (_outNode)
	{
		deletedData = _outNode->nodeDataModel()->outData(_outPortIndex);
		if (_inNode && deletedData) {
			deletedData->setConnectionType(CntType::Break);
		}
	}

	propagateData(deletedData);
}