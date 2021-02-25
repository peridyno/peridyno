#include "BlockState.h"

#include "QtBlockDataModel.h"

#include "QtConnection.h"

using QtNodes::BlockState;
using QtNodes::BlockDataType;
using QtNodes::QtBlockDataModel;
using QtNodes::PortType;
using QtNodes::PortIndex;
using QtNodes::QtConnection;

BlockState::
BlockState(std::unique_ptr<QtBlockDataModel> const &model)
  : _inConnections(model->nPorts(PortType::In))
  , _outConnections(model->nPorts(PortType::Out))
  , _reaction(NOT_REACTING)
  , _reactingPortType(PortType::None)
  , _resizing(false)
{}


std::vector<BlockState::ConnectionPtrSet> const &
BlockState::
getEntries(PortType portType) const
{
  if (portType == PortType::In)
    return _inConnections;
  else
    return _outConnections;
}


std::vector<BlockState::ConnectionPtrSet> &
BlockState::
getEntries(PortType portType)
{
  if (portType == PortType::In)
    return _inConnections;
  else
    return _outConnections;
}


BlockState::ConnectionPtrSet
BlockState::
connections(PortType portType, PortIndex portIndex) const
{
  auto const &connections = getEntries(portType);

  return connections[portIndex];
}


void
BlockState::
setConnection(PortType portType,
              PortIndex portIndex,
              QtConnection& connection)
{
  auto &connections = getEntries(portType);

  connections.at(portIndex).insert(std::make_pair(connection.id(),
                                               &connection));
}


void
BlockState::
eraseConnection(PortType portType,
                PortIndex portIndex,
                QUuid id)
{
  getEntries(portType)[portIndex].erase(id);
}


BlockState::ReactToConnectionState
BlockState::
reaction() const
{
  return _reaction;
}


PortType
BlockState::
reactingPortType() const
{
  return _reactingPortType;
}


BlockDataType
BlockState::
reactingDataType() const
{
  return _reactingDataType;
}


void
BlockState::
setReaction(ReactToConnectionState reaction,
            PortType reactingPortType,
            BlockDataType reactingDataType)
{
  _reaction = reaction;

  _reactingPortType = reactingPortType;

  _reactingDataType = std::move(reactingDataType);
}


bool
BlockState::
isReacting() const
{
  return _reaction == REACTING;
}


void
BlockState::
setResizing(bool resizing)
{
  _resizing = resizing;
}


bool
BlockState::
resizing() const
{
  return _resizing;
}
