#include "ConnectionInteraction.h"

#include "QtConnectionGraphicsObject.h"
#include "QtBlockGraphicsObject.h"
#include "QtBlockDataModel.h"
#include "DataModelRegistry.h"
#include "QtFlowScene.h"

using QtNodes::ConnectionInteraction;
using QtNodes::PortType;
using QtNodes::PortIndex;
using QtNodes::QtFlowScene;
using QtNodes::QtBlock;
using QtNodes::QtConnection;
using QtNodes::QtBlockDataModel;
using QtNodes::TypeConverter;


ConnectionInteraction::
ConnectionInteraction(QtBlock& node, QtConnection& connection, QtFlowScene& scene)
  : _block(&node)
  , _connection(&connection)
  , _scene(&scene)
{}


bool
ConnectionInteraction::
canConnect(PortIndex &portIndex, TypeConverter & converter) const
{
	// 1) Connection requires a port
	PortType requiredPort = connectionRequiredPort();

	if (requiredPort == PortType::None)
	{
		return false;
	}

	// 1.5) Forbid connecting the node to itself
	PortType start_porttype = oppositePort(requiredPort);
	QtBlock* start_block = _connection->getBlock(oppositePort(requiredPort));

	if (start_block == _block)
		return false;

	// 2) connection point is on top of the node port

	QPointF connectionPoint = connectionEndScenePosition(requiredPort);

	portIndex = nodePortIndexUnderScenePoint(requiredPort,
		connectionPoint);

	if (portIndex == INVALID)
	{
		return false;
	}

	// Check whether the output data can be accepted by the input block
	auto start_portIndex =  _connection->getPortIndex(oppositePort(requiredPort));
	auto start_data = start_block->nodeDataModel()->portData(start_porttype, start_portIndex);

	auto target_data = _block->nodeDataModel()->portData(requiredPort, portIndex);

	if (!target_data->isKindOf(*start_data))
	{
		return false;
	}


	// 3) Node port is vacant

	// port should be empty
	if (!isNodePortAccesible(requiredPort, portIndex))
		return false;

	// 4) Connection type equals node port type, or there is a registered type conversion that can translate between the two

	auto connectionDataType =
		_connection->dataType(oppositePort(requiredPort));

	auto const   &modelTarget = _block->nodeDataModel();
	BlockDataType candidateNodeDataType = modelTarget->dataType(requiredPort, portIndex);

	if (connectionDataType.id != candidateNodeDataType.id)
	{
		if (requiredPort == PortType::In)
		{
			converter = _scene->registry().getTypeConverter(connectionDataType, candidateNodeDataType);
		}
		else if (requiredPort == PortType::Out)
		{
			converter = _scene->registry().getTypeConverter(candidateNodeDataType, connectionDataType);
		}

		return (converter != nullptr);
	}

	return true;
}


bool
ConnectionInteraction::
tryConnect() const
{
  // 1) Check conditions from 'canConnect'
  PortIndex portIndex = INVALID;

  TypeConverter converter;

  if (!canConnect(portIndex, converter))
  {
    return false;
  }

  // 1.5) If the connection is possible but a type conversion is needed,
  //      assign a convertor to connection
  if (converter)
  {
    _connection->setTypeConverter(converter);
  }

  // 2) Assign node to required port in Connection
  PortType requiredPort = connectionRequiredPort();
  _block->nodeState().setConnection(requiredPort,
                                   portIndex,
                                   *_connection);

  // 3) Assign Connection to empty port in NodeState
  // The port is not longer required after this function
  _connection->setNodeToPort(*_block, requiredPort, portIndex);

  // 4) Adjust Connection geometry

  _block->nodeGraphicsObject().moveConnections();

  // 5) Poke model to initiate data transfer

  auto outNode = _connection->getBlock(PortType::Out);
  if (outNode)
  {
    PortIndex outPortIndex = _connection->getPortIndex(PortType::Out);
    outNode->onDataUpdated(outPortIndex);
  }

  return true;
}


/// 1) Node and Connection should be already connected
/// 2) If so, clear Connection entry in the NodeState
/// 3) Set Connection end to 'requiring a port'
bool
ConnectionInteraction::
disconnect(PortType portToDisconnect) const
{
  PortIndex portIndex =
    _connection->getPortIndex(portToDisconnect);

  BlockState &state = _block->nodeState();

  // clear pointer to Connection in the NodeState
  state.getEntries(portToDisconnect)[portIndex].clear();

  // 4) Propagate invalid data to IN node
  _connection->propagateDeletedData();

  // clear Connection side
  _connection->clearNode(portToDisconnect);

  _connection->setRequiredPort(portToDisconnect);

  _connection->getConnectionGraphicsObject().grabMouse();

  return true;
}


// ------------------ util functions below

PortType
ConnectionInteraction::
connectionRequiredPort() const
{
  auto const &state = _connection->connectionState();

  return state.requiredPort();
}


QPointF
ConnectionInteraction::
connectionEndScenePosition(PortType portType) const
{
  auto &go =
    _connection->getConnectionGraphicsObject();

  ConnectionGeometry& geometry = _connection->connectionGeometry();

  QPointF endPoint = geometry.getEndPoint(portType);

  return go.mapToScene(endPoint);
}


QPointF
ConnectionInteraction::
nodePortScenePosition(PortType portType, PortIndex portIndex) const
{
  BlockGeometry const &geom = _block->nodeGeometry();

  QPointF p = geom.portScenePosition(portIndex, portType);

  QtBlockGraphicsObject& ngo = _block->nodeGraphicsObject();

  return ngo.sceneTransform().map(p);
}


PortIndex
ConnectionInteraction::
nodePortIndexUnderScenePoint(PortType portType,
                             QPointF const & scenePoint) const
{
  BlockGeometry const &nodeGeom = _block->nodeGeometry();

  QTransform sceneTransform =
    _block->nodeGraphicsObject().sceneTransform();

  PortIndex portIndex = nodeGeom.checkHitScenePoint(portType,
                                                    scenePoint,
                                                    sceneTransform);
  return portIndex;
}


bool
ConnectionInteraction::
isNodePortAccesible(PortType portType, PortIndex portIndex) const
{
  BlockState const & nodeState = _block->nodeState();

  auto const & entries = nodeState.getEntries(portType);

  if (entries[portIndex].empty()) return true;

  if (portType == PortType::Out)
  {
	  const auto outPolicy = _block->nodeDataModel()->portOutConnectionPolicy(portIndex);
	  return outPolicy == QtBlockDataModel::ConnectionPolicy::Many;
  }
  else
  {
	  const auto inPolicy = _block->nodeDataModel()->portInConnectionPolicy(portIndex);
	  return inPolicy == QtBlockDataModel::ConnectionPolicy::Many;
  }
  
  
}
