#pragma once

#include <QtCore/QUuid>
#include <QtWidgets/QGraphicsScene>

#include <unordered_map>
#include <tuple>
#include <functional>

#include "QUuidStdHash.h"
#include "Export.h"
#include "DataModelRegistry.h"
#include "TypeConverter.h"
#include "memory.h"

#include "Node.h"

namespace QtNodes
{

class QtBlockDataModel;
class FlowItemInterface;
class QtBlock;
class QtBlockGraphicsObject;
class QtConnection;
class QtConnectionGraphicsObject;
class BlockStyle;

using dyno::Node;

/// Scene holds connections and nodes.
class NODE_EDITOR_PUBLIC QtFlowScene
  : public QGraphicsScene
{
  Q_OBJECT
public:

  QtFlowScene(std::shared_ptr<DataModelRegistry> registry,
            QObject * parent = Q_NULLPTR);

  QtFlowScene(QObject * parent = Q_NULLPTR);

  ~QtFlowScene();

public:

  std::shared_ptr<QtConnection>
  createConnection(PortType connectedPort,
                   QtBlock& node,
                   PortIndex portIndex);

  std::shared_ptr<QtConnection>
  createConnection(QtBlock& nodeIn,
                   PortIndex portIndexIn,
                   QtBlock& nodeOut,
                   PortIndex portIndexOut,
                   TypeConverter const & converter = TypeConverter{});

  std::shared_ptr<QtConnection> restoreConnection(QJsonObject const &connectionJson);

  void deleteConnection(QtConnection& connection);

  QtBlock&createNode(std::unique_ptr<QtBlockDataModel> && dataModel);

  QtBlock&restoreNode(QJsonObject const& nodeJson);

  void removeNode(QtBlock& node);

  DataModelRegistry&registry() const;

  void setRegistry(std::shared_ptr<DataModelRegistry> registry);

  void iterateOverNodes(std::function<void(QtBlock*)> const & visitor);

  void iterateOverNodeData(std::function<void(QtBlockDataModel*)> const & visitor);

  void iterateOverNodeDataDependentOrder(std::function<void(QtBlockDataModel*)> const & visitor);

  QPointF getNodePosition(QtBlock const& node) const;

  void setNodePosition(QtBlock& node, QPointF const& pos) const;

  QSizeF getNodeSize(QtBlock const& node) const;

public:

  std::unordered_map<QUuid, std::unique_ptr<QtBlock> > const & nodes() const;

  std::unordered_map<QUuid, std::shared_ptr<QtConnection> > const & connections() const;

  std::vector<QtBlock*> allNodes() const;

  std::vector<QtBlock*> selectedNodes() const;

public:

  void clearScene();

  void newNode();

  void save() const;

  void load();

  QByteArray saveToMemory() const;

  void loadFromMemory(const QByteArray& data);

Q_SIGNALS:

	/**
	* @brief Node has been created but not on the scene yet.
	* @see nodePlaced()
	*/
	void nodeCreated(QtBlock &n);

	/**
	* @brief Node has been added to the scene.
	* @details Connect to this signal if need a correct position of node.
	* @see nodeCreated()
	*/
	void nodePlaced(QtBlock &n);

	void nodeDeleted(QtBlock &n);

	void connectionCreated(QtConnection const &c);
	void connectionDeleted(QtConnection const &c);

	void nodeMoved(QtBlock& n, const QPointF& newLocation);

	void nodeSelected(QtBlock& n);

	void nodeDoubleClicked(QtBlock& n);

	void connectionHovered(QtConnection& c, QPoint screenPos);

	void nodeHovered(QtBlock& n, QPoint screenPos);

	void connectionHoverLeft(QtConnection& c);

	void nodeHoverLeft(QtBlock& n);

	void nodeContextMenu(QtBlock& n, const QPointF& pos);

private:

	using SharedConnection = std::shared_ptr<QtConnection>;
	using UniqueBlock = std::unique_ptr<QtBlock>;

	std::unordered_map<QUuid, SharedConnection> _connections;
	std::unordered_map<QUuid, UniqueBlock>       _nodes;
	std::shared_ptr<DataModelRegistry>          _registry;


private Q_SLOTS:

  void setupConnectionSignals(QtConnection const& c);

  void sendConnectionCreatedToNodes(QtConnection const& c);
  void sendConnectionDeletedToNodes(QtConnection const& c);


};

QtBlock*
locateNodeAt(QPointF scenePoint, QtFlowScene &scene,
             QTransform const & viewTransform);
}
