#include "ConnectionState.h"

#include <iostream>

#include <QtCore/QPointF>

#include "QtFlowScene.h"
#include "QtBlock.h"

using QtNodes::ConnectionState;
using QtNodes::QtBlock;

ConnectionState::
~ConnectionState()
{
  resetLastHoveredNode();
}


void
ConnectionState::
interactWithNode(QtBlock* node)
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


void
ConnectionState::
setLastHoveredNode(QtBlock* node)
{
  _lastHoveredNode = node;
}


void
ConnectionState::
resetLastHoveredNode()
{
  if (_lastHoveredNode)
    _lastHoveredNode->resetReactionToConnection();

  _lastHoveredNode = nullptr;
}
