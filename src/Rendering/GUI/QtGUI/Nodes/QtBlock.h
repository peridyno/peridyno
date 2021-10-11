#pragma once


#include <QtCore/QObject>
#include <QtCore/QUuid>

#include <QtCore/QJsonObject>

#include "PortType.h"

#include "Export.h"
#include "BlockState.h"
#include "BlockGeometry.h"
#include "BlockData.h"
#include "QtBlockGraphicsObject.h"
#include "QtConnectionGraphicsObject.h"
#include "Serializable.h"
#include "memory.h"

namespace QtNodes
{

class QtConnection;
class ConnectionState;
class QtBlockGraphicsObject;
class QtBlockDataModel;

class NODE_EDITOR_PUBLIC QtBlock
  : public QObject
  , public Serializable
{
  Q_OBJECT

public:

  /// NodeDataModel should be an rvalue and is moved into the Node
  QtBlock(std::unique_ptr<QtBlockDataModel> && dataModel);

  virtual
  ~QtBlock();

public:

  QJsonObject
  save() const override;

  void
  restore(QJsonObject const &json) override;

public:

  QUuid
  id() const;

  void reactToPossibleConnection(PortType,
                                 BlockDataType const &,
                                 QPointF const & scenePoint);

  void
  resetReactionToConnection();

public:

  QtBlockGraphicsObject const &
  nodeGraphicsObject() const;

  QtBlockGraphicsObject &
  nodeGraphicsObject();

  void
  setGraphicsObject(std::unique_ptr<QtBlockGraphicsObject>&& graphics);

  BlockGeometry&
  nodeGeometry();

  BlockGeometry const&
  nodeGeometry() const;

  BlockState const &
  nodeState() const;

  BlockState &
  nodeState();

  QtBlockDataModel*
  nodeDataModel() const;

public Q_SLOTS: // data propagation

  /// Propagates incoming data to the underlying model.
  void
  propagateData(std::shared_ptr<BlockData> nodeData,
                PortIndex inPortIndex) const;

  /// Fetches data from model's OUT #index port
  /// and propagates it to the connection
  void
  onDataUpdated(PortIndex index);

  /// update the graphic part if the size of the embeddedwidget changes
  void
  onNodeSizeUpdated();

private:

  // addressing

  QUuid _uid;

  // data

  std::unique_ptr<QtBlockDataModel> _nodeDataModel;

  BlockState _nodeState;

  // painting

  BlockGeometry _nodeGeometry;

  std::unique_ptr<QtBlockGraphicsObject> _nodeGraphicsObject;
};
}
