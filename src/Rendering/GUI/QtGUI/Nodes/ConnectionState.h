#pragma once

#include <QtCore/QUuid>

#include "PortType.h"

class QPointF;

namespace QtNodes
{

class QtBlock;

/// Stores currently draggind end.
/// Remembers last hovered Node.
class ConnectionState
{
public:

  ConnectionState(PortType port = PortType::None)
    : _requiredPort(port)
  {}

  ConnectionState(const ConnectionState&) = delete;
  ConnectionState operator=(const ConnectionState&) = delete;

  ~ConnectionState();

public:

  void setRequiredPort(PortType end)
  { _requiredPort = end; }

  PortType requiredPort() const
  { return _requiredPort; }

  bool requiresPort() const
  { return _requiredPort != PortType::None; }

  void setNoRequiredPort()
  { _requiredPort = PortType::None; }

public:

  void interactWithNode(QtBlock* node);

  void setLastHoveredNode(QtBlock* node);

  QtBlock*
  lastHoveredNode() const
  { return _lastHoveredNode; }

  void resetLastHoveredNode();

private:

  PortType _requiredPort;

  QtBlock* _lastHoveredNode{nullptr};
};
}
