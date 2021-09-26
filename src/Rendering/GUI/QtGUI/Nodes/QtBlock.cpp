#include "QtBlock.h"

#include <QtCore/QObject>

#include <utility>
#include <iostream>

#include "QtFlowScene.h"

#include "QtBlockGraphicsObject.h"
#include "QtBlockDataModel.h"

#include "QtConnectionGraphicsObject.h"
#include "ConnectionState.h"

using QtNodes::QtBlock;
using QtNodes::BlockGeometry;
using QtNodes::BlockState;
using QtNodes::BlockData;
using QtNodes::BlockDataType;
using QtNodes::QtBlockDataModel;
using QtNodes::QtBlockGraphicsObject;
using QtNodes::PortIndex;
using QtNodes::PortType;

QtBlock::
QtBlock(std::unique_ptr<QtBlockDataModel> && dataModel)
  : _uid(QUuid::createUuid())
  , _nodeDataModel(std::move(dataModel))
  , _nodeState(_nodeDataModel)
  , _nodeGeometry(_nodeDataModel)
  , _nodeGraphicsObject(nullptr)
{
  _nodeGeometry.recalculateSize();

  // propagate data: model => node
  connect(_nodeDataModel.get(), &QtBlockDataModel::dataUpdated,
          this, &QtBlock::onDataUpdated);

  connect(_nodeDataModel.get(), &QtBlockDataModel::embeddedWidgetSizeUpdated,
          this, &QtBlock::onNodeSizeUpdated );
}


QtBlock::
~QtBlock() = default;

QJsonObject
QtBlock::
save() const
{
  QJsonObject nodeJson;

  nodeJson["id"] = _uid.toString();

  nodeJson["model"] = _nodeDataModel->save();

  QJsonObject obj;
  obj["x"] = _nodeGraphicsObject->pos().x();
  obj["y"] = _nodeGraphicsObject->pos().y();
  nodeJson["position"] = obj;

  return nodeJson;
}


void
QtBlock::
restore(QJsonObject const& json)
{
  _uid = QUuid(json["id"].toString());

  QJsonObject positionJson = json["position"].toObject();
  QPointF     point(positionJson["x"].toDouble(),
                    positionJson["y"].toDouble());
  _nodeGraphicsObject->setPos(point);

  _nodeDataModel->restore(json["model"].toObject());
}


QUuid
QtBlock::
id() const
{
  return _uid;
}


void
QtBlock::
reactToPossibleConnection(PortType reactingPortType,
                          BlockDataType const &reactingDataType,
                          QPointF const &scenePoint)
{
  QTransform const t = _nodeGraphicsObject->sceneTransform();

  QPointF p = t.inverted().map(scenePoint);

  _nodeGeometry.setDraggingPosition(p);

  _nodeGraphicsObject->update();

  _nodeState.setReaction(BlockState::REACTING,
                         reactingPortType,
                         reactingDataType);
}


void
QtBlock::
resetReactionToConnection()
{
  _nodeState.setReaction(BlockState::NOT_REACTING);
  _nodeGraphicsObject->update();
}


QtBlockGraphicsObject const &
QtBlock::
nodeGraphicsObject() const
{
  return *_nodeGraphicsObject.get();
}


QtBlockGraphicsObject &
QtBlock::
nodeGraphicsObject()
{
  return *_nodeGraphicsObject.get();
}


void
QtBlock::
setGraphicsObject(std::unique_ptr<QtBlockGraphicsObject>&& graphics)
{
  _nodeGraphicsObject = std::move(graphics);

  _nodeGeometry.recalculateSize();
}


BlockGeometry&
QtBlock::
nodeGeometry()
{
  return _nodeGeometry;
}


BlockGeometry const&
QtBlock::
nodeGeometry() const
{
  return _nodeGeometry;
}


BlockState const &
QtBlock::
nodeState() const
{
  return _nodeState;
}


BlockState &
QtBlock::
nodeState()
{
  return _nodeState;
}


QtBlockDataModel*
QtBlock::
nodeDataModel() const
{
  return _nodeDataModel.get();
}


void
QtBlock::
propagateData(std::shared_ptr<BlockData> nodeData,
              PortIndex inPortIndex) const
{
  _nodeDataModel->setInData(std::move(nodeData), inPortIndex);

  //Recalculate the nodes visuals. A data change can result in the node taking more space than before, so this forces a recalculate+repaint on the affected node
  _nodeGraphicsObject->setGeometryChanged();
  _nodeGeometry.recalculateSize();
  _nodeGraphicsObject->update();
  _nodeGraphicsObject->moveConnections();
}


void
QtBlock::
onDataUpdated(PortIndex index)
{
  auto nodeData = _nodeDataModel->outData(index);

  auto connections =
    _nodeState.connections(PortType::Out, index);

  for (auto const & c : connections)
    c.second->propagateData(nodeData);
}

void
QtBlock::
onNodeSizeUpdated()
{
    if( nodeDataModel()->embeddedWidget() )
    {
        nodeDataModel()->embeddedWidget()->adjustSize();
    }
    nodeGeometry().recalculateSize();
    for(PortType type: {PortType::In, PortType::Out})
    {
        for(auto& conn_set : nodeState().getEntries(type))
        {
            for(auto& pair: conn_set)
            {
                QtConnection* conn = pair.second;
                conn->getConnectionGraphicsObject().move();
            }
        }
    }
}
