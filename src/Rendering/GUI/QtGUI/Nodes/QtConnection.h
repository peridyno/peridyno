#pragma once

#include <QtCore/QObject>
#include <QtCore/QUuid>
#include <QtCore/QVariant>

#include "PortType.h"
#include "BlockData.h"

#include "Serializable.h"
#include "ConnectionState.h"
#include "ConnectionGeometry.h"
#include "TypeConverter.h"
#include "QUuidStdHash.h"
#include "Export.h"
#include "memory.h"

class QPointF;

namespace QtNodes
{

class QtBlock;
class BlockData;
class QtConnectionGraphicsObject;

///
class NODE_EDITOR_PUBLIC QtConnection
  : public QObject
  , public Serializable
{

  Q_OBJECT

public:

  /// New Connection is attached to the port of the given Node.
  /// The port has parameters (portType, portIndex).
  /// The opposite connection end will require anothre port.
  QtConnection(PortType portType,
             QtBlock& node,
             PortIndex portIndex);

  QtConnection(QtBlock& nodeIn,
             PortIndex portIndexIn,
             QtBlock& nodeOut,
             PortIndex portIndexOut,
             TypeConverter converter =
               TypeConverter{});

  QtConnection(const QtConnection&) = delete;
  QtConnection operator=(const QtConnection&) = delete;

  ~QtConnection();

public:

  QJsonObject
  save() const override;

public:

  QUuid
  id() const;

  /// Remembers the end being dragged.
  /// Invalidates Node address.
  /// Grabs mouse.
  void
  setRequiredPort(PortType portType);
  PortType
  requiredPort() const;

  void
  setGraphicsObject(std::unique_ptr<QtConnectionGraphicsObject>&& graphics);

  /// Assigns a node to the required port.
  /// It is assumed that there is a required port, no extra checks
  void
  setNodeToPort(QtBlock& node,
                PortType portType,
                PortIndex portIndex);

  void
  removeFromNodes() const;

public:

  QtConnectionGraphicsObject&
  getConnectionGraphicsObject() const;

  ConnectionState const &
  connectionState() const;
  ConnectionState&
  connectionState();

  ConnectionGeometry&
  connectionGeometry();

  ConnectionGeometry const&
  connectionGeometry() const;

  QtBlock*
  getBlock(PortType portType) const;

  QtBlock*&
  getBlock(PortType portType);

  PortIndex
  getPortIndex(PortType portType) const;

  void
  clearNode(PortType portType);

  BlockDataType
  dataType(PortType portType) const;

  void
  setTypeConverter(TypeConverter converter);

  bool
  complete() const;

public: // data propagation

  void
  propagateData(std::shared_ptr<BlockData> nodeData) const;

  void
  propagateEmptyData() const;

  void propagateDeletedData() const;

Q_SIGNALS:

  void
  connectionCompleted(QtConnection const&) const;

  void
  connectionMadeIncomplete(QtConnection const&) const;

private:

  QUuid _uid;

private:

  QtBlock* _outNode = nullptr;
  QtBlock* _inNode  = nullptr;

  PortIndex _outPortIndex;
  PortIndex _inPortIndex;

private:

  ConnectionState    _connectionState;
  ConnectionGeometry _connectionGeometry;

  std::unique_ptr<QtConnectionGraphicsObject>_connectionGraphicsObject;

  TypeConverter _converter;

Q_SIGNALS:

  void
  updated(QtConnection& conn) const;
};
}
