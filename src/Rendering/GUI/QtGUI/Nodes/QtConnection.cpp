#include "QtConnection.h"

#include <cmath>
#include <utility>

#include <QtWidgets/QtWidgets>
#include <QtGlobal>

#include "QtBlock.h"
#include "QtFlowScene.h"
#include "QtFlowView.h"

#include "BlockGeometry.h"
#include "QtBlockGraphicsObject.h"
#include "QtBlockDataModel.h"

#include "ConnectionState.h"
#include "ConnectionGeometry.h"
#include "QtConnectionGraphicsObject.h"

using QtNodes::QtConnection;
using QtNodes::PortType;
using QtNodes::PortIndex;
using QtNodes::ConnectionState;
using QtNodes::QtBlock;
using QtNodes::BlockData;
using QtNodes::BlockDataType;
using QtNodes::QtConnectionGraphicsObject;
using QtNodes::ConnectionGeometry;
using QtNodes::TypeConverter;

QtConnection::
QtConnection(PortType portType,
           QtBlock& node,
           PortIndex portIndex)
  : _uid(QUuid::createUuid())
  , _outPortIndex(INVALID)
  , _inPortIndex(INVALID)
  , _connectionState()
{
  setNodeToPort(node, portType, portIndex);

  setRequiredPort(oppositePort(portType));
}


QtConnection::
QtConnection(QtBlock& nodeIn,
           PortIndex portIndexIn,
           QtBlock& nodeOut,
           PortIndex portIndexOut,
           TypeConverter typeConverter)
  : _uid(QUuid::createUuid())
  , _outNode(&nodeOut)
  , _inNode(&nodeIn)
  , _outPortIndex(portIndexOut)
  , _inPortIndex(portIndexIn)
  , _connectionState()
  , _converter(std::move(typeConverter))
{
  setNodeToPort(nodeIn, PortType::In, portIndexIn);
  setNodeToPort(nodeOut, PortType::Out, portIndexOut);
}


QtConnection::
~QtConnection()
{
  if (complete()) connectionMadeIncomplete(*this);

  this->propagateDeletedData();

  if (_inNode)
  {
    _inNode->nodeGraphicsObject().update();
  }

  if (_outNode)
  {
    _outNode->nodeGraphicsObject().update();
  }
}


QJsonObject
QtConnection::
save() const
{
  QJsonObject connectionJson;

  if (_inNode && _outNode)
  {
    connectionJson["in_id"] = _inNode->id().toString();
    connectionJson["in_index"] = _inPortIndex;

    connectionJson["out_id"] = _outNode->id().toString();
    connectionJson["out_index"] = _outPortIndex;

    if (_converter)
    {
      auto getTypeJson = [this](PortType type)
      {
        QJsonObject typeJson;
        BlockDataType nodeType = this->dataType(type);
        typeJson["id"] = nodeType.id;
        typeJson["name"] = nodeType.name;

        return typeJson;
      };

      QJsonObject converterTypeJson;

      converterTypeJson["in"] = getTypeJson(PortType::In);
      converterTypeJson["out"] = getTypeJson(PortType::Out);

      connectionJson["converter"] = converterTypeJson;
    }
  }

  return connectionJson;
}


QUuid
QtConnection::
id() const
{
  return _uid;
}


bool
QtConnection::
complete() const
{
  return _inNode != nullptr && _outNode != nullptr;
}


void
QtConnection::
setRequiredPort(PortType dragging)
{
  _connectionState.setRequiredPort(dragging);

  switch (dragging)
  {
    case PortType::Out:
      _outNode      = nullptr;
      _outPortIndex = INVALID;
      break;

    case PortType::In:
      _inNode      = nullptr;
      _inPortIndex = INVALID;
      break;

    default:
      break;
  }
}


PortType
QtConnection::
requiredPort() const
{
  return _connectionState.requiredPort();
}


void
QtConnection::
setGraphicsObject(std::unique_ptr<QtConnectionGraphicsObject>&& graphics)
{
  _connectionGraphicsObject = std::move(graphics);

  // This function is only called when the ConnectionGraphicsObject
  // is newly created. At this moment both end coordinates are (0, 0)
  // in Connection G.O. coordinates. The position of the whole
  // Connection G. O. in scene coordinate system is also (0, 0).
  // By moving the whole object to the Node Port position
  // we position both connection ends correctly.

  if (requiredPort() != PortType::None)
  {

    PortType attachedPort = oppositePort(requiredPort());

    PortIndex attachedPortIndex = getPortIndex(attachedPort);

    auto node = getBlock(attachedPort);

    QTransform nodeSceneTransform =
      node->nodeGraphicsObject().sceneTransform();

    QPointF pos = node->nodeGeometry().portScenePosition(attachedPortIndex,
                                                         attachedPort,
                                                         nodeSceneTransform);

    _connectionGraphicsObject->setPos(pos);
  }

  _connectionGraphicsObject->move();
}



PortIndex
QtConnection::
getPortIndex(PortType portType) const
{
  PortIndex result = INVALID;

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


void
QtConnection::
setNodeToPort(QtBlock& node,
              PortType portType,
              PortIndex portIndex)
{
  bool wasIncomplete = !complete();

  auto& nodeWeak = getBlock(portType);

  nodeWeak = &node;

  if (portType == PortType::Out)
    _outPortIndex = portIndex;
  else
    _inPortIndex = portIndex;

  _connectionState.setNoRequiredPort();

  updated(*this);
  if (complete() && wasIncomplete) {
    connectionCompleted(*this);
  }
}


void
QtConnection::
removeFromNodes() const
{
  if (_inNode)
    _inNode->nodeState().eraseConnection(PortType::In, _inPortIndex, id());

  if (_outNode)
    _outNode->nodeState().eraseConnection(PortType::Out, _outPortIndex, id());
}


QtConnectionGraphicsObject&
QtConnection::
getConnectionGraphicsObject() const
{
  return *_connectionGraphicsObject;
}


ConnectionState&
QtConnection::
connectionState()
{
  return _connectionState;
}


ConnectionState const&
QtConnection::
connectionState() const
{
  return _connectionState;
}


ConnectionGeometry&
QtConnection::
connectionGeometry()
{
  return _connectionGeometry;
}


ConnectionGeometry const&
QtConnection::
connectionGeometry() const
{
  return _connectionGeometry;
}


QtBlock*
QtConnection::
getBlock(PortType portType) const
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


QtBlock*&
QtConnection::
getBlock(PortType portType)
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
  Q_UNREACHABLE();
}


void
QtConnection::
clearNode(PortType portType)
{
  if (complete()) {
    connectionMadeIncomplete(*this);
  }

  getBlock(portType) = nullptr;

  if (portType == PortType::In)
    _inPortIndex = INVALID;
  else
    _outPortIndex = INVALID;
}


BlockDataType
QtConnection::
dataType(PortType portType) const
{
  if (_inNode && _outNode)
  {
    auto const & model = (portType == PortType::In) ?
                        _inNode->nodeDataModel() :
                        _outNode->nodeDataModel();
    PortIndex index = (portType == PortType::In) ? 
                      _inPortIndex :
                      _outPortIndex;

    return model->dataType(portType, index);
  }
  else 
  {
    QtBlock* validNode;
    PortIndex index = INVALID;

    if ((validNode = _inNode))
    {
      index    = _inPortIndex;
      portType = PortType::In;
    }
    else if ((validNode = _outNode))
    {
      index    = _outPortIndex;
      portType = PortType::Out;
    }

    if (validNode)
    {
      auto const &model = validNode->nodeDataModel();

      return model->dataType(portType, index);
    }
  }

  Q_UNREACHABLE();
}


void
QtConnection::
setTypeConverter(TypeConverter converter)
{
  _converter = std::move(converter);
}


void
QtConnection::
propagateData(std::shared_ptr<BlockData> nodeData) const
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


void
QtConnection::
propagateEmptyData() const
{
  std::shared_ptr<BlockData> emptyData;

  propagateData(emptyData);
}

void QtNodes::QtConnection::propagateDeletedData() const
{
	std::shared_ptr<BlockData> deletedData;
	if (_outNode)
	{
		deletedData = _outNode->nodeDataModel()->outData(_outPortIndex);
		if (_inNode)
		{
			deletedData->setDisconnected(true);
		}
	}

	propagateData(deletedData);
}
