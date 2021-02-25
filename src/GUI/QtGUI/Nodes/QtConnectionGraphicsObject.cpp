#include "QtConnectionGraphicsObject.h"

#include <QtWidgets/QGraphicsSceneMouseEvent>
#include <QtWidgets/QGraphicsDropShadowEffect>
#include <QtWidgets/QGraphicsBlurEffect>
#include <QtWidgets/QStyleOptionGraphicsItem>
#include <QtWidgets/QGraphicsView>

#include "QtFlowScene.h"

#include "QtConnection.h"
#include "ConnectionGeometry.h"
#include "ConnectionPainter.h"
#include "ConnectionState.h"
#include "QtConnectionBlurEffect.h"

#include "QtBlockGraphicsObject.h"

#include "ConnectionInteraction.h"

#include "QtBlock.h"

using QtNodes::QtConnectionGraphicsObject;
using QtNodes::QtConnection;
using QtNodes::QtFlowScene;

QtConnectionGraphicsObject::
QtConnectionGraphicsObject(QtFlowScene &scene,
                         QtConnection &connection)
  : _scene(scene)
  , _connection(connection)
{
  _scene.addItem(this);

  setFlag(QGraphicsItem::ItemIsMovable, true);
  setFlag(QGraphicsItem::ItemIsFocusable, true);
  setFlag(QGraphicsItem::ItemIsSelectable, true);

  setAcceptHoverEvents(true);

  // addGraphicsEffect();

  setZValue(-1.0);
}


QtConnectionGraphicsObject::
~QtConnectionGraphicsObject()
{
  _scene.removeItem(this);
}


QtNodes::QtConnection&
QtConnectionGraphicsObject::
connection()
{
  return _connection;
}


QRectF
QtConnectionGraphicsObject::
boundingRect() const
{
  return _connection.connectionGeometry().boundingRect();
}


QPainterPath
QtConnectionGraphicsObject::
shape() const
{
#ifdef DEBUG_DRAWING

  //QPainterPath path;

  //path.addRect(boundingRect());
  //return path;

#else
  auto const &geom =
    _connection.connectionGeometry();

  return ConnectionPainter::getPainterStroke(geom);

#endif
}


void
QtConnectionGraphicsObject::
setGeometryChanged()
{
  prepareGeometryChange();
}


void
QtConnectionGraphicsObject::
move()
{
  for(PortType portType: { PortType::In, PortType::Out } )
  {
    if (auto node = _connection.getBlock(portType))
    {
      auto const &nodeGraphics = node->nodeGraphicsObject();

      auto const &nodeGeom = node->nodeGeometry();

      QPointF scenePos =
        nodeGeom.portScenePosition(_connection.getPortIndex(portType),
                                   portType,
                                   nodeGraphics.sceneTransform());

      QTransform sceneTransform = this->sceneTransform();

      QPointF connectionPos = sceneTransform.inverted().map(scenePos);

      _connection.connectionGeometry().setEndPoint(portType,
                                                   connectionPos);

      _connection.getConnectionGraphicsObject().setGeometryChanged();
      _connection.getConnectionGraphicsObject().update();
    }
  }

}

void QtConnectionGraphicsObject::lock(bool locked)
{
  setFlag(QGraphicsItem::ItemIsMovable, !locked);
  setFlag(QGraphicsItem::ItemIsFocusable, !locked);
  setFlag(QGraphicsItem::ItemIsSelectable, !locked);
}


void
QtConnectionGraphicsObject::
paint(QPainter* painter,
      QStyleOptionGraphicsItem const* option,
      QWidget*)
{
  painter->setClipRect(option->exposedRect);

  ConnectionPainter::paint(painter,
                           _connection);
}


void
QtConnectionGraphicsObject::
mousePressEvent(QGraphicsSceneMouseEvent* event)
{
  QGraphicsItem::mousePressEvent(event);
  //event->ignore();
}


void
QtConnectionGraphicsObject::
mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
  prepareGeometryChange();

  auto view = static_cast<QGraphicsView*>(event->widget());
  auto node = locateNodeAt(event->scenePos(),
                           _scene,
                           view->transform());

  auto &state = _connection.connectionState();

  state.interactWithNode(node);
  if (node)
  {
    node->reactToPossibleConnection(state.requiredPort(),
                                    _connection.dataType(oppositePort(state.requiredPort())),
                                    event->scenePos());
  }

  //-------------------

  QPointF offset = event->pos() - event->lastPos();

  auto requiredPort = _connection.requiredPort();

  if (requiredPort != PortType::None)
  {
    _connection.connectionGeometry().moveEndPoint(requiredPort, offset);
  }

  //-------------------

  update();

  event->accept();
}


void
QtConnectionGraphicsObject::
mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
  ungrabMouse();
  event->accept();

  auto node = locateNodeAt(event->scenePos(), _scene,
                           _scene.views()[0]->transform());

  ConnectionInteraction interaction(*node, _connection, _scene);

  if (node && interaction.tryConnect())
  {
    node->resetReactionToConnection();
  }

  if (_connection.connectionState().requiresPort())
  {
    _scene.deleteConnection(_connection);
  }
}


void
QtConnectionGraphicsObject::
hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
  _connection.connectionGeometry().setHovered(true);

  update();
  _scene.connectionHovered(connection(), event->screenPos());
  event->accept();
}


void
QtConnectionGraphicsObject::
hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
  _connection.connectionGeometry().setHovered(false);

  update();
  _scene.connectionHoverLeft(connection());
  event->accept();
}


void
QtConnectionGraphicsObject::
addGraphicsEffect()
{
  auto effect = new QGraphicsBlurEffect;

  effect->setBlurRadius(5);
  setGraphicsEffect(effect);

  //auto effect = new QGraphicsDropShadowEffect;
  //auto effect = new ConnectionBlurEffect(this);
  //effect->setOffset(4, 4);
  //effect->setColor(QColor(Qt::gray).darker(800));
}
