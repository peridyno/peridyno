#include "QtBlockGraphicsObject.h"

#include <iostream>
#include <cstdlib>

#include <QtWidgets/QtWidgets>
#include <QtWidgets/QGraphicsEffect>

#include "QtConnectionGraphicsObject.h"
#include "ConnectionState.h"

#include "QtFlowScene.h"
#include "BlockPainter.h"

#include "QtBlock.h"
#include "QtBlockDataModel.h"
#include "ConnectionInteraction.h"

#include "StyleCollection.h"

using QtNodes::QtBlockGraphicsObject;
using QtNodes::QtBlock;
using QtNodes::QtFlowScene;

QtBlockGraphicsObject::
QtBlockGraphicsObject(QtFlowScene &scene,
                   QtBlock& node)
  : _scene(scene)
  , _node(node)
  , _locked(false)
  , _proxyWidget(nullptr)
{
  _scene.addItem(this);

  setFlag(QGraphicsItem::ItemDoesntPropagateOpacityToChildren, true);
  setFlag(QGraphicsItem::ItemIsMovable, true);
  setFlag(QGraphicsItem::ItemIsFocusable, true);
  setFlag(QGraphicsItem::ItemIsSelectable, true);
  setFlag(QGraphicsItem::ItemSendsScenePositionChanges, true);

  setCacheMode( QGraphicsItem::DeviceCoordinateCache );

  auto const &nodeStyle = node.nodeDataModel()->nodeStyle();

  {
    auto effect = new QGraphicsDropShadowEffect;
    effect->setOffset(4, 4);
    effect->setBlurRadius(20);
    effect->setColor(nodeStyle.ShadowColor);

    setGraphicsEffect(effect);
  }

  setOpacity(nodeStyle.Opacity);

  setAcceptHoverEvents(true);

  setZValue(0);

  embedQWidget();

  // connect to the move signals to emit the move signals in FlowScene
  auto onMoveSlot = [this] {
    _scene.nodeMoved(_node, pos());
  };
  connect(this, &QGraphicsObject::xChanged, this, onMoveSlot);
  connect(this, &QGraphicsObject::yChanged, this, onMoveSlot);
}


QtBlockGraphicsObject::
~QtBlockGraphicsObject()
{
  _scene.removeItem(this);
}


QtBlock&
QtBlockGraphicsObject::
node()
{
  return _node;
}


QtBlock const&
QtBlockGraphicsObject::
node() const
{
  return _node;
}


void
QtBlockGraphicsObject::
embedQWidget()
{
  BlockGeometry & geom = _node.nodeGeometry();

  if (auto w = _node.nodeDataModel()->embeddedWidget())
  {
    _proxyWidget = new QGraphicsProxyWidget(this);

    _proxyWidget->setWidget(w);

    _proxyWidget->setPreferredWidth(5);

    geom.recalculateSize();

    if (w->sizePolicy().verticalPolicy() & QSizePolicy::ExpandFlag)
    {
      // If the widget wants to use as much vertical space as possible, set it to have the geom's equivalentWidgetHeight.
      _proxyWidget->setMinimumHeight(geom.equivalentWidgetHeight());
    }

    _proxyWidget->setPos(geom.widgetPosition());

    update();

    _proxyWidget->setOpacity(1.0);
    _proxyWidget->setFlag(QGraphicsItem::ItemIgnoresParentOpacity);
  }
}


QRectF
QtBlockGraphicsObject::
boundingRect() const
{
  return _node.nodeGeometry().boundingRect();
}


void
QtBlockGraphicsObject::
setGeometryChanged()
{
  prepareGeometryChange();
}


void
QtBlockGraphicsObject::
moveConnections() const
{
  BlockState const & nodeState = _node.nodeState();

  for (PortType portType: {PortType::In, PortType::Out})
  {
    auto const & connectionEntries =
      nodeState.getEntries(portType);

    for (auto const & connections : connectionEntries)
    {
      for (auto & con : connections)
        con.second->getConnectionGraphicsObject().move();
    }
  }
}


void
QtBlockGraphicsObject::
lock(bool locked)
{
  _locked = locked;

  setFlag(QGraphicsItem::ItemIsMovable, !locked);
  setFlag(QGraphicsItem::ItemIsFocusable, !locked);
  setFlag(QGraphicsItem::ItemIsSelectable, !locked);
}


void
QtBlockGraphicsObject::
paint(QPainter * painter,
      QStyleOptionGraphicsItem const* option,
      QWidget* )
{
  painter->setClipRect(option->exposedRect);

  BlockPainter::paint(painter, _node, _scene);
}


QVariant
QtBlockGraphicsObject::
itemChange(GraphicsItemChange change, const QVariant &value)
{
  if (change == ItemPositionChange && scene())
  {
    moveConnections();
  }

  return QGraphicsItem::itemChange(change, value);
}


void
QtBlockGraphicsObject::
mousePressEvent(QGraphicsSceneMouseEvent * event)
{
	if (_locked)
		return;

	// deselect all other items after this one is selected
	if (!isSelected() &&
		!(event->modifiers() & Qt::ControlModifier))
	{
		_scene.clearSelection();
	}

	for (PortType portToCheck : {PortType::In, PortType::Out})
	{
		BlockGeometry const & nodeGeometry = _node.nodeGeometry();

		// TODO do not pass sceneTransform
		int const portIndex = nodeGeometry.checkHitScenePoint(portToCheck,
			event->scenePos(),
			sceneTransform());

		if (portIndex != INVALID)
		{
			BlockState const & nodeState = _node.nodeState();

			std::unordered_map<QUuid, QtConnection*> connections =
				nodeState.connections(portToCheck, portIndex);

			// dragging connection for the in port
			if (portToCheck == PortType::In)
			{
				auto const inPolicy = _node.nodeDataModel()->portInConnectionPolicy(portIndex);

				//If only one input is required, disconnect the previous connection first
				if (!connections.empty() && 
					inPolicy == QtBlockDataModel::ConnectionPolicy::One)
				{
					auto con = connections.begin()->second;

					ConnectionInteraction interaction(_node, *con, _scene);

					interaction.disconnect(portToCheck);
				}
				else
				{
					//Create a new connection starting from a input port
					auto connection = _scene.createConnection(portToCheck,
						_node,
						portIndex);

					_node.nodeState().setConnection(portToCheck,
						portIndex,
						*connection);

					connection->getConnectionGraphicsObject().grabMouse();
				}
			}
			else // initialize new Connection
			{
				auto const outPolicy = _node.nodeDataModel()->portOutConnectionPolicy(portIndex);
				if (!connections.empty() &&
					outPolicy == QtBlockDataModel::ConnectionPolicy::One)
				{
					_scene.deleteConnection(*connections.begin()->second);
				}

				////Create a new connection starting from a output port
				auto connection = _scene.createConnection(portToCheck,
					_node,
					portIndex);

				_node.nodeState().setConnection(portToCheck,
					portIndex,
					*connection);

				connection->getConnectionGraphicsObject().grabMouse();
			}
		}
	}

	auto pos = event->pos();
	auto & geom = _node.nodeGeometry();
	auto & state = _node.nodeState();

	if (_node.nodeDataModel()->resizable() &&
		geom.resizeRect().contains(QPoint(pos.x(),
			pos.y())))
	{
		state.setResizing(true);
	}

}


void
QtBlockGraphicsObject::
mouseMoveEvent(QGraphicsSceneMouseEvent * event)
{
  auto & geom  = _node.nodeGeometry();
  auto & state = _node.nodeState();

  if (state.resizing())
  {
    auto diff = event->pos() - event->lastPos();

    if (auto w = _node.nodeDataModel()->embeddedWidget())
    {
      prepareGeometryChange();

      auto oldSize = w->size();

      oldSize += QSize(diff.x(), diff.y());

      w->setFixedSize(oldSize);

      _proxyWidget->setMinimumSize(oldSize);
      _proxyWidget->setMaximumSize(oldSize);
      _proxyWidget->setPos(geom.widgetPosition());

      geom.recalculateSize();
      update();

      moveConnections();

      event->accept();
    }
  }
  else
  {
    QGraphicsObject::mouseMoveEvent(event);

    if (event->lastPos() != event->pos())
      moveConnections();

    event->ignore();
  }

  QRectF r = scene()->sceneRect();

  r = r.united(mapToScene(boundingRect()).boundingRect());

  scene()->setSceneRect(r);
}


void
QtBlockGraphicsObject::
mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
  auto & state = _node.nodeState();

  state.setResizing(false);

  QGraphicsObject::mouseReleaseEvent(event);

  if (isSelected())
  {
	  _scene.nodeSelected(node());
  }

  // position connections precisely after fast node move
  moveConnections();
}


void
QtBlockGraphicsObject::
hoverEnterEvent(QGraphicsSceneHoverEvent * event)
{
  // bring all the colliding nodes to background
  QList<QGraphicsItem *> overlapItems = collidingItems();

  for (QGraphicsItem *item : overlapItems)
  {
    if (item->zValue() > 0.0)
    {
      item->setZValue(0.0);
    }
  }

  // bring this node forward
  setZValue(1.0);

  _node.nodeGeometry().setHovered(true);
  update();
  _scene.nodeHovered(node(), event->screenPos());
  event->accept();
}


void
QtBlockGraphicsObject::
hoverLeaveEvent(QGraphicsSceneHoverEvent * event)
{
  _node.nodeGeometry().setHovered(false);
  update();
  _scene.nodeHoverLeft(node());
  event->accept();
}


void
QtBlockGraphicsObject::
hoverMoveEvent(QGraphicsSceneHoverEvent * event)
{
  auto pos    = event->pos();
  auto & geom = _node.nodeGeometry();

  if (_node.nodeDataModel()->resizable() &&
      geom.resizeRect().contains(QPoint(pos.x(), pos.y())))
  {
    setCursor(QCursor(Qt::SizeFDiagCursor));
  }
  else
  {
    setCursor(QCursor());
  }

  event->accept();
}


void
QtBlockGraphicsObject::
mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
  QGraphicsItem::mouseDoubleClickEvent(event);

  _scene.nodeDoubleClicked(node());
}


void
QtBlockGraphicsObject::
contextMenuEvent(QGraphicsSceneContextMenuEvent* event)
{
  _scene.nodeContextMenu(node(), mapToScene(event->pos()));
}
