#pragma once

#include <vector>
#include <unordered_map>

#include <QtCore/QUuid>

#include "Export.h"

#include "PortType.h"
#include "BlockData.h"
#include "memory.h"

namespace QtNodes
{

class QtConnection;
class QtBlockDataModel;

/// Contains vectors of connected input and output connections.
/// Stores bool for reacting on hovering connections
class NODE_EDITOR_PUBLIC BlockState
{
public:
  enum ReactToConnectionState
  {
    REACTING,
    NOT_REACTING
  };

public:

  BlockState(std::unique_ptr<QtBlockDataModel> const &model);

public:

  using ConnectionPtrSet =
          std::unordered_map<QUuid, QtConnection*>;

  /// Returns vector of connections ID.
  /// Some of them can be empty (null)
  std::vector<ConnectionPtrSet> const&
  getEntries(PortType) const;

  std::vector<ConnectionPtrSet> &
  getEntries(PortType);

  ConnectionPtrSet
  connections(PortType portType, PortIndex portIndex) const;

  void
  setConnection(PortType portType,
                PortIndex portIndex,
                QtConnection& connection);

  void
  eraseConnection(PortType portType,
                  PortIndex portIndex,
                  QUuid id);

  ReactToConnectionState
  reaction() const;

  PortType
  reactingPortType() const;

  BlockDataType
  reactingDataType() const;

  void
  setReaction(ReactToConnectionState reaction,
              PortType reactingPortType = PortType::None,

              BlockDataType reactingDataType =
                BlockDataType());

  bool
  isReacting() const;

  void
  setResizing(bool resizing);

  bool
  resizing() const;

private:

  std::vector<ConnectionPtrSet> _inConnections;
  std::vector<ConnectionPtrSet> _outConnections;

  ReactToConnectionState _reaction;
  PortType     _reactingPortType;
  BlockDataType _reactingDataType;

  bool _resizing;
};
}
