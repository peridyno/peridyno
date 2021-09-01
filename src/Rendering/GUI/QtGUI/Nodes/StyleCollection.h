#pragma once

#include "BlockStyle.h"
#include "ConnectionStyle.h"
#include "FlowViewStyle.h"

namespace QtNodes
{

class StyleCollection
{
public:

  static
  BlockStyle const&
  nodeStyle();

  static
  ConnectionStyle const&
  connectionStyle();

  static
  FlowViewStyle const&
  flowViewStyle();

public:

  static
  void
  setNodeStyle(BlockStyle);

  static
  void
  setConnectionStyle(ConnectionStyle);

  static
  void
  setFlowViewStyle(FlowViewStyle);

private:

  StyleCollection() = default;

  StyleCollection(StyleCollection const&) = delete;

  StyleCollection&
  operator=(StyleCollection const&) = delete;

  static
  StyleCollection&
  instance();

private:

  BlockStyle _nodeStyle;

  ConnectionStyle _connectionStyle;

  FlowViewStyle _flowViewStyle;
};
}
