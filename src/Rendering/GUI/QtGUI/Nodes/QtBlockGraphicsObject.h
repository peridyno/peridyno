#pragma once

#include <QtCore/QUuid>
#include <QtWidgets/QGraphicsObject>

#include "QtConnection.h"

#include "BlockGeometry.h"
#include "BlockState.h"

class QGraphicsProxyWidget;

namespace QtNodes
{

class QtFlowScene;
class FlowItemEntry;

/// Class reacts on GUI events, mouse clicks and
/// forwards painting operation.
class QtBlockGraphicsObject : public QGraphicsObject
{
  Q_OBJECT

public:
  QtBlockGraphicsObject(QtFlowScene &scene,
                     QtBlock& node);

  virtual
  ~QtBlockGraphicsObject();

  QtBlock&
  node();

  QtBlock const&
  node() const;

  QRectF
  boundingRect() const override;

  void
  setGeometryChanged();

  /// Visits all attached connections and corrects
  /// their corresponding end points.
  void
  moveConnections() const;

  enum { Type = UserType + 1 };

  int
  type() const override { return Type; }

  void
  lock(bool locked);

protected:
  void
  paint(QPainter*                       painter,
        QStyleOptionGraphicsItem const* option,
        QWidget*                        widget = 0) override;

  QVariant
  itemChange(GraphicsItemChange change, const QVariant &value) override;

  void
  mousePressEvent(QGraphicsSceneMouseEvent* event) override;

  void
  mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;

  void
  mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

  void
  hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;

  void
  hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;

  void
  hoverMoveEvent(QGraphicsSceneHoverEvent *) override;

  void
  mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) override;

  void
  contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;

private:
  void
  embedQWidget();

private:

  QtFlowScene & _scene;

  QtBlock& _node;

  bool _locked;

  // either nullptr or owned by parent QGraphicsItem
  QGraphicsProxyWidget * _proxyWidget;
};
}
