#include "QtFlowScene.h"

#include <stdexcept>
#include <utility>

#include <QtWidgets/QGraphicsSceneMoveEvent>
#include <QtWidgets/QFileDialog>
#include <QtCore/QByteArray>
#include <QtCore/QBuffer>
#include <QtCore/QDataStream>
#include <QtCore/QFile>

#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonArray>
#include <QtCore/QtGlobal>
#include <QtCore/QDebug>
#include <QtWidgets/QMessageBox>

#include "QtBlock.h"
#include "QtBlockGraphicsObject.h"

#include "QtBlockGraphicsObject.h"
#include "QtConnectionGraphicsObject.h"

#include "QtConnection.h"

#include "QtFlowView.h"
#include "DataModelRegistry.h"

using QtNodes::QtFlowScene;
using QtNodes::QtBlock;
using QtNodes::QtBlockGraphicsObject;
using QtNodes::QtConnection;
using QtNodes::DataModelRegistry;
using QtNodes::QtBlockDataModel;
using QtNodes::PortType;
using QtNodes::PortIndex;
using QtNodes::TypeConverter;


QtFlowScene::QtFlowScene(std::shared_ptr<DataModelRegistry> registry,
          QObject * parent)
  : QGraphicsScene(parent)
  , _registry(std::move(registry))
{
  setItemIndexMethod(QGraphicsScene::NoIndex);

  // This connection should come first
  connect(this, &QtFlowScene::connectionCreated, this, &QtFlowScene::setupConnectionSignals);
  connect(this, &QtFlowScene::connectionCreated, this, &QtFlowScene::sendConnectionCreatedToNodes);
  connect(this, &QtFlowScene::connectionDeleted, this, &QtFlowScene::sendConnectionDeletedToNodes);
}

QtFlowScene::QtFlowScene(QObject * parent)
	: QtFlowScene(std::make_shared<DataModelRegistry>(), parent)
{
}


QtFlowScene::~QtFlowScene()
{
  clearScene();
}


//------------------------------------------------------------------------------

std::shared_ptr<QtConnection>
QtFlowScene::createConnection(PortType connectedPort,
                 QtBlock& node,
                 PortIndex portIndex)
{
  auto connection = std::make_shared<QtConnection>(connectedPort, node, portIndex);

  auto cgo = detail::make_unique<QtConnectionGraphicsObject>(*this, *connection);

  // after this function connection points are set to node port
  connection->setGraphicsObject(std::move(cgo));

  _connections[connection->id()] = connection;

  // Note: this connection isn't truly created yet. It's only partially created.
  // Thus, don't send the connectionCreated(...) signal.

  connect(connection.get(),
          &QtConnection::connectionCompleted,
          this,
          [this](QtConnection const& c) {
            connectionCreated(c);
          });

  return connection;
}


std::shared_ptr<QtConnection>
QtFlowScene::createConnection(QtBlock& nodeIn,
									 PortIndex portIndexIn,
									 QtBlock& nodeOut,
									 PortIndex portIndexOut,
									 TypeConverter const &converter)
{
  auto connection =
    std::make_shared<QtConnection>(nodeIn,
                                 portIndexIn,
                                 nodeOut,
                                 portIndexOut,
                                 converter);

  auto cgo = detail::make_unique<QtConnectionGraphicsObject>(*this, *connection);

  nodeIn.nodeState().setConnection(PortType::In, portIndexIn, *connection);
  nodeOut.nodeState().setConnection(PortType::Out, portIndexOut, *connection);

  // after this function connection points are set to node port
  connection->setGraphicsObject(std::move(cgo));

  // trigger data propagation
  nodeOut.onDataUpdated(portIndexOut);

  _connections[connection->id()] = connection;

  connectionCreated(*connection);

  return connection;
}


std::shared_ptr<QtConnection>
QtFlowScene::
restoreConnection(QJsonObject const &connectionJson)
{
  QUuid nodeInId  = QUuid(connectionJson["in_id"].toString());
  QUuid nodeOutId = QUuid(connectionJson["out_id"].toString());

  PortIndex portIndexIn  = connectionJson["in_index"].toInt();
  PortIndex portIndexOut = connectionJson["out_index"].toInt();

  auto nodeIn  = _nodes[nodeInId].get();
  auto nodeOut = _nodes[nodeOutId].get();

  auto getConverter = [&]()
  {
    QJsonValue converterVal = connectionJson["converter"];

    if (!converterVal.isUndefined())
    {
      QJsonObject converterJson = converterVal.toObject();

      BlockDataType inType { converterJson["in"].toObject()["id"].toString(),
                            converterJson["in"].toObject()["name"].toString() };

      BlockDataType outType { converterJson["out"].toObject()["id"].toString(),
                             converterJson["out"].toObject()["name"].toString() };

      auto converter  =
        registry().getTypeConverter(outType, inType);

      if (converter)
        return converter;
    }

    return TypeConverter{};
  };

  std::shared_ptr<QtConnection> connection =
    createConnection(*nodeIn, portIndexIn,
                     *nodeOut, portIndexOut,
                     getConverter());

  // Note: the connectionCreated(...) signal has already been sent
  // by createConnection(...)

  return connection;
}


void
QtFlowScene::
deleteConnection(QtConnection& connection)
{
  auto it = _connections.find(connection.id());
  if (it != _connections.end()) {
    connection.removeFromNodes();
    _connections.erase(it);
  }
}


QtBlock&
QtFlowScene::
createNode(std::unique_ptr<QtBlockDataModel> && dataModel)
{
  auto node = detail::make_unique<QtBlock>(std::move(dataModel));
  auto ngo  = detail::make_unique<QtBlockGraphicsObject>(*this, *node);

  node->setGraphicsObject(std::move(ngo));

  auto nodePtr = node.get();
  _nodes[node->id()] = std::move(node);

  nodeCreated(*nodePtr);
  return *nodePtr;
}


QtBlock&
QtFlowScene::
restoreNode(QJsonObject const& nodeJson)
{
  QString modelName = nodeJson["model"].toObject()["name"].toString();

  auto dataModel = registry().create(modelName);

  if (!dataModel)
    throw std::logic_error(std::string("No registered model with name ") +
                           modelName.toLocal8Bit().data());

  auto node = detail::make_unique<QtBlock>(std::move(dataModel));
  auto ngo  = detail::make_unique<QtBlockGraphicsObject>(*this, *node);
  node->setGraphicsObject(std::move(ngo));

  node->restore(nodeJson);

  auto nodePtr = node.get();
  _nodes[node->id()] = std::move(node);

  nodePlaced(*nodePtr);
  nodeCreated(*nodePtr);
  return *nodePtr;
}


void
QtFlowScene::
removeNode(QtBlock& node)
{
  // call signal
  nodeDeleted(node);

  for(auto portType: {PortType::In,PortType::Out})
  {
    auto nodeState = node.nodeState();
    auto const & nodeEntries = nodeState.getEntries(portType);

    for (auto &connections : nodeEntries)
    {
      for (auto const &pair : connections)
        deleteConnection(*pair.second);
    }
  }

  _nodes.erase(node.id());
}


DataModelRegistry&
QtFlowScene::
registry() const
{
  return *_registry;
}


void
QtFlowScene::
setRegistry(std::shared_ptr<DataModelRegistry> registry)
{
  _registry = std::move(registry);
}


void
QtFlowScene::
iterateOverNodes(std::function<void(QtBlock*)> const & visitor)
{
  for (const auto& _node : _nodes)
  {
    visitor(_node.second.get());
  }
}


void
QtFlowScene::
iterateOverNodeData(std::function<void(QtBlockDataModel*)> const & visitor)
{
  for (const auto& _node : _nodes)
  {
    visitor(_node.second->nodeDataModel());
  }
}


void
QtFlowScene::
iterateOverNodeDataDependentOrder(std::function<void(QtBlockDataModel*)> const & visitor)
{
  std::set<QUuid> visitedNodesSet;

  //A leaf node is a node with no input ports, or all possible input ports empty
  auto isNodeLeaf =
    [](QtBlock const &node, QtBlockDataModel const &model)
    {
      for (unsigned int i = 0; i < model.nPorts(PortType::In); ++i)
      {
        auto connections = node.nodeState().connections(PortType::In, i);
        if (!connections.empty())
        {
          return false;
        }
      }

      return true;
    };

  //Iterate over "leaf" nodes
  for (auto const &_node : _nodes)
  {
    auto const &node = _node.second;
    auto model       = node->nodeDataModel();

    if (isNodeLeaf(*node, *model))
    {
      visitor(model);
      visitedNodesSet.insert(node->id());
    }
  }

  auto areNodeInputsVisitedBefore =
    [&](QtBlock const &node, QtBlockDataModel const &model)
    {
      for (size_t i = 0; i < model.nPorts(PortType::In); ++i)
      {
        auto connections = node.nodeState().connections(PortType::In, i);

        for (auto& conn : connections)
        {
          if (visitedNodesSet.find(conn.second->getBlock(PortType::Out)->id()) == visitedNodesSet.end())
          {
            return false;
          }
        }
      }

      return true;
    };

  //Iterate over dependent nodes
  while (_nodes.size() != visitedNodesSet.size())
  {
    for (auto const &_node : _nodes)
    {
      auto const &node = _node.second;
      if (visitedNodesSet.find(node->id()) != visitedNodesSet.end())
        continue;

      auto model = node->nodeDataModel();

      if (areNodeInputsVisitedBefore(*node, *model))
      {
        visitor(model);
        visitedNodesSet.insert(node->id());
      }
    }
  }
}


QPointF
QtFlowScene::
getNodePosition(const QtBlock& node) const
{
  return node.nodeGraphicsObject().pos();
}


void
QtFlowScene::
setNodePosition(QtBlock& node, const QPointF& pos) const
{
  node.nodeGraphicsObject().setPos(pos);
  node.nodeGraphicsObject().moveConnections();
}


QSizeF
QtFlowScene::
getNodeSize(const QtBlock& node) const
{
  return QSizeF(node.nodeGeometry().width(), node.nodeGeometry().height());
}


std::unordered_map<QUuid, std::unique_ptr<QtBlock> > const &
QtFlowScene::
nodes() const
{
  return _nodes;
}


std::unordered_map<QUuid, std::shared_ptr<QtConnection> > const &
QtFlowScene::
connections() const
{
  return _connections;
}


std::vector<QtBlock*>
QtFlowScene::
allNodes() const
{
  std::vector<QtBlock*> nodes;

  std::transform(_nodes.begin(),
                 _nodes.end(),
                 std::back_inserter(nodes),
                 [](std::pair<QUuid const, std::unique_ptr<QtBlock>> const & p) { return p.second.get(); });

  return nodes;
}


std::vector<QtBlock*>
QtFlowScene::
selectedNodes() const
{
  QList<QGraphicsItem*> graphicsItems = selectedItems();

  std::vector<QtBlock*> ret;
  ret.reserve(graphicsItems.size());

  for (QGraphicsItem* item : graphicsItems)
  {
    auto ngo = qgraphicsitem_cast<QtBlockGraphicsObject*>(item);

    if (ngo != nullptr)
    {
      ret.push_back(&ngo->node());
    }
  }

  return ret;
}


//------------------------------------------------------------------------------

void
QtFlowScene::
clearScene()
{
  //Manual node cleanup. Simply clearing the holding datastructures doesn't work, the code crashes when
  // there are both nodes and connections in the scene. (The data propagation internal logic tries to propagate
  // data through already freed connections.)
  while (_connections.size() > 0)
  {
    deleteConnection( *_connections.begin()->second );
  }

  while (_nodes.size() > 0)
  {
    removeNode( *_nodes.begin()->second );
  }
}


void
QtFlowScene::
save() const
{
  QString fileName =
    QFileDialog::getSaveFileName(nullptr,
                                 tr("Open Flow Scene"),
                                 QDir::homePath(),
                                 tr("Flow Scene Files (*.flow)"));

  if (!fileName.isEmpty())
  {
    if (!fileName.endsWith("flow", Qt::CaseInsensitive))
      fileName += ".flow";

    QFile file(fileName);
    if (file.open(QIODevice::WriteOnly))
    {
      file.write(saveToMemory());
    }
  }
}


void
QtFlowScene::
load()
{
  clearScene();

  //-------------

  QString fileName =
    QFileDialog::getOpenFileName(nullptr,
                                 tr("Open Flow Scene"),
                                 QDir::homePath(),
                                 tr("Flow Scene Files (*.flow)"));

  if (!QFileInfo::exists(fileName))
    return;

  QFile file(fileName);

  if (!file.open(QIODevice::ReadOnly))
    return;

  QByteArray wholeFile = file.readAll();

  loadFromMemory(wholeFile);
}


QByteArray
QtFlowScene::
saveToMemory() const
{
  QJsonObject sceneJson;

  QJsonArray nodesJsonArray;

  for (auto const & pair : _nodes)
  {
    auto const &node = pair.second;

    nodesJsonArray.append(node->save());
  }

  sceneJson["nodes"] = nodesJsonArray;

  QJsonArray connectionJsonArray;
  for (auto const & pair : _connections)
  {
    auto const &connection = pair.second;

    QJsonObject connectionJson = connection->save();

    if (!connectionJson.isEmpty())
      connectionJsonArray.append(connectionJson);
  }

  sceneJson["connections"] = connectionJsonArray;

  QJsonDocument document(sceneJson);

  return document.toJson();
}


void
QtFlowScene::
loadFromMemory(const QByteArray& data)
{
  QJsonObject const jsonDocument = QJsonDocument::fromJson(data).object();

  QJsonArray nodesJsonArray = jsonDocument["nodes"].toArray();

  for (QJsonValueRef node : nodesJsonArray)
  {
    restoreNode(node.toObject());
  }

  QJsonArray connectionJsonArray = jsonDocument["connections"].toArray();

  for (QJsonValueRef connection : connectionJsonArray)
  {
    restoreConnection(connection.toObject());
  }
}


void
QtFlowScene::
setupConnectionSignals(QtConnection const& c)
{
  connect(&c,
          &QtConnection::connectionMadeIncomplete,
          this,
          &QtFlowScene::connectionDeleted,
          Qt::UniqueConnection);
}


void
QtFlowScene::
sendConnectionCreatedToNodes(QtConnection const& c)
{
  QtBlock* from = c.getBlock(PortType::Out);
  QtBlock* to   = c.getBlock(PortType::In);

  Q_ASSERT(from != nullptr);
  Q_ASSERT(to != nullptr);

  from->nodeDataModel()->outputConnectionCreated(c);
  to->nodeDataModel()->inputConnectionCreated(c);
}


void
QtFlowScene::
sendConnectionDeletedToNodes(QtConnection const& c)
{
  QtBlock* from = c.getBlock(PortType::Out);
  QtBlock* to   = c.getBlock(PortType::In);

  Q_ASSERT(from != nullptr);
  Q_ASSERT(to != nullptr);

  from->nodeDataModel()->outputConnectionDeleted(c);
  to->nodeDataModel()->inputConnectionDeleted(c);
}


//------------------------------------------------------------------------------
namespace QtNodes
{

QtBlock*
locateNodeAt(QPointF scenePoint, QtFlowScene &scene,
             QTransform const & viewTransform)
{
  // items under cursor
  QList<QGraphicsItem*> items =
    scene.items(scenePoint,
                Qt::IntersectsItemShape,
                Qt::DescendingOrder,
                viewTransform);

  //// items convertable to NodeGraphicsObject
  std::vector<QGraphicsItem*> filteredItems;

  std::copy_if(items.begin(),
               items.end(),
               std::back_inserter(filteredItems),
               [] (QGraphicsItem * item)
    {
      return (dynamic_cast<QtBlockGraphicsObject*>(item) != nullptr);
    });

  QtBlock* resultNode = nullptr;

  if (!filteredItems.empty())
  {
    QGraphicsItem* graphicsItem = filteredItems.front();
    auto ngo = dynamic_cast<QtBlockGraphicsObject*>(graphicsItem);

    resultNode = &ngo->node();
  }

  return resultNode;
}


void QtFlowScene::newNode()
{
/* 	dyno::SceneGraph& scene = dyno::SceneGraph::getInstance();
	auto root = scene.getRootNode();

	root->removeAllChildren();

	std::shared_ptr<dyno::ParticleElasticBody<dyno::DataType3f>> bunny = std::make_shared<dyno::ParticleElasticBody<dyno::DataType3f>>();
	root->addChild(bunny);
	//	bunny->getRenderModule()->setColor(Vec3f(0, 1, 1));
	bunny->setMass(1.0);
	bunny->loadParticles("../../data/bunny/bunny_points.obj");
	bunny->loadSurface("../../data/bunny/bunny_mesh.obj");
	bunny->translate(dyno::Vec3f(0.5, 0.2, 0.5));
	bunny->setVisible(true);

// 	auto renderer = std::make_shared<dyno::PVTKSurfaceMeshRender>();
// 	renderer->setName("VTK Mesh Renderer");
// 	bunny->getSurfaceNode()->addVisualModule(renderer);

	auto pRenderer = std::make_shared<dyno::PVTKPointSetRender>();
	pRenderer->setName("VTK Point Renderer");
	bunny->addVisualModule(pRenderer);

	scene.invalid();
	scene.initialize();

	auto mlist = bunny->getModuleList();

	auto c = bunny->getAnimationPipeline()->entry();

	std::map<std::string, QtBlock*> moduleMap;

	int mSize = bunny->getAnimationPipeline()->size();


	auto addModuleWidget = [&](Module* m) -> void
	{
		auto module_name = m->getName();

		auto type = std::make_unique<QtModuleWidget>(m);

		auto& node = this->createNode(std::move(type));

		moduleMap[module_name] = &node;

		QPointF posView;

		node.nodeGraphicsObject().setPos(posView);

		this->nodePlaced(node);
	};

	addModuleWidget(bunny->getMechanicalState().get());

	for (; c != bunny->getAnimationPipeline()->finished(); c++)
	{
		addModuleWidget(c.get());
	}

	auto createModuleConnections = [&](Module* m) -> void
	{
		auto out_node = moduleMap[m->getName()];

		auto fields = m->getOutputFields();

		for (int i = 0; i < fields.size(); i++)
		{
			auto sink_fields = fields[i]->getSinkFields();
			for (int j = 0; j < sink_fields.size(); j++)
			{
				auto in_module = dynamic_cast<Module*>(sink_fields[j]->getParent());
				if (in_module != nullptr)
				{
					auto in_fields = in_module->getInputFields();

					int in_port = -1;
					for (int t = 0; t < in_fields.size(); t++)
					{
						if (sink_fields[j] == in_fields[t])
						{
							in_port = t;
							break;
						}
					}

					if (in_port != -1)
					{
						auto in_node = moduleMap[in_module->getName()];

						createConnection(*in_node, in_port, *out_node, i);
					}
				}
			}
		}
	};


	createModuleConnections(bunny->getMechanicalState().get());
	c = bunny->getAnimationPipeline()->entry();
	for (; c != bunny->getAnimationPipeline()->finished(); c++)
	{
		createModuleConnections(c.get());
	}*/
}
}
