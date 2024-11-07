#include "WtFlowScene.h"

WtFlowScene::WtFlowScene(std::shared_ptr<WtDataModelRegistry> registry)
	: _registry(registry)
{
	//setItemIndexMethod(QGraphicsScene::NoIndex);
	//// This connection should come first
	//connect(this, &QtFlowScene::connectionCreated, this, &QtFlowScene::setupConnectionSignals);
	//connect(this, &QtFlowScene::connectionCreated, this, &QtFlowScene::sendConnectionCreatedToNodes);
	//connect(this, &QtFlowScene::connectionDeleted, this, &QtFlowScene::sendConnectionDeletedToNodes);
}

WtFlowScene::WtFlowScene()
{
}

WtFlowScene::~WtFlowScene()
{
	clearScene();
}

std::shared_ptr<WtConnection> WtFlowScene::createConnection(
	PortType connectedPort,
	WtNode& node,
	PortIndex portIndex,
	Wt::WPainter* painter)
{
	auto connection = std::make_shared<WtConnection>(connectedPort, node, portIndex);

	auto cgo = detail::make_unique<WtConnectionGraphicsObject>(*this, *connection, painter);

	// after this function connection points are set to node port
	connection->setGraphicsObject(std::move(cgo));

	_connections[connection->id()] = connection;

	// Note: this connection isn't truly created yet. It's only partially created.
	// Thus, don't send the connectionCreated(...) signal.

	/*connect(connection.get(),
		&QtConnection::connectionCompleted,
		this,
		[this](QtConnection const& c) {
			connectionCreated(c);
		});*/

	return connection;
}

std::shared_ptr<WtConnection> WtFlowScene::createConnection(
	WtNode& nodeIn,
	PortIndex portIndexIn,
	WtNode& nodeOut,
	PortIndex portIndexOut,
	Wt::WPainter* painter,
	TypeConverter const& converter)
{
	auto connection = std::make_shared<WtConnection>(
		nodeIn,
		portIndexIn,
		nodeOut,
		portIndexOut,
		converter);

	auto cgo = detail::make_unique<WtConnectionGraphicsObject>(*this, *connection, painter);

	nodeIn.nodeState().setConnection(PortType::In, portIndexIn, *connection);

	nodeOut.nodeState().setConnection(PortType::Out, portIndexOut, *connection);

	// after this function connection points are set to node port
	connection->setGraphicsObject(std::move(cgo));

	// trigger data propagation
	nodeOut.onDataUpdated(portIndexOut);

	_connections[connection->id()] = connection;

	//signal
	//connectionCreated(*connection);

	return connection;
}

void WtFlowScene::deleteConnection(WtConnection& connection)
{
	auto it = _connections.find(connection.id());
	if (it != _connections.end())
	{
		connection.removeFromNodes();
		_connections.erase(it);
	}
}

WtNode& WtFlowScene::createNode(std::unique_ptr<WtNodeDataModel>&& dataModel, Wt::WPainter* painter, bool isSelected)
{
	Wt::WPaintDevice* paintDevice = painter->device();

	auto node = detail::make_unique<WtNode>(std::move(dataModel), paintDevice);
	auto ngo = detail::make_unique<WtNodeGraphicsObject>(*this, *node, painter, isSelected);

	node->setGraphicsObject(std::move(ngo));

	auto nodePtr = node.get();
	_nodes[node->id()] = std::move(node);

	return *nodePtr;
}

WtDataModelRegistry& WtFlowScene::registry() const
{
	return *_registry;
}

void WtFlowScene::setRegistry(std::shared_ptr<WtDataModelRegistry> registry)
{
	_registry = std::move(registry);
}

void WtFlowScene::iterateOverNodes(std::function<void(WtNode*)> const& visitor)
{
	for (const auto& _node : _nodes)
	{
		visitor(_node.second.get());
	}
}

void WtFlowScene::iterateOverNodeData(std::function<void(WtNodeDataModel*)> const& visitor)
{
	for (const auto& _node : _nodes)
	{
		visitor(_node.second->nodeDataModel());
	}
}

void WtFlowScene::iterateOverNodeDataDependentOrder(std::function<void(WtNodeDataModel*)> const& visitor)
{
	std::set<Wt::Guid> visitedNodesSet;

	auto isNodeLeaf = [](WtNode const& node, WtNodeDataModel const& model)
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
	for (auto const& _node : _nodes)
	{
		auto const& node = _node.second;
		auto model = node->nodeDataModel();

		if (isNodeLeaf(*node, *model))
		{
			visitor(model);
			visitedNodesSet.insert(node->id());
		}
	}

	auto areNodeInputsVisitedBefore =
		[&](WtNode const& node, WtNodeDataModel const& model)
		{
			for (size_t i = 0; i < model.nPorts(PortType::In); ++i)
			{
				auto connections = node.nodeState().connections(PortType::In, i);

				for (auto& conn : connections)
				{
					/*if (visitedNodesSet.find(conn.second->getNode(PortType::Out)->id()) == visitedNodesSet.end())
					{
						return false;
					}*/
					return false;
				}
			}

			return true;
		};

	//Iterate over dependent nodes
	while (_nodes.size() != visitedNodesSet.size())
	{
		for (auto const& _node : _nodes)
		{
			auto const& node = _node.second;
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

Wt::WPointF WtFlowScene::getNodePosition(const WtNode& node) const
{
	return node.nodeGraphicsObject().getPos();
}

//Wt::WPointF WtFlowScene::setNodePosition(WtNode& node, const Wt::WPointF& pos) const
//{
//	//node.nodeGraphicsObject().setPos(pos);
//	node.nodeGraphicsObject().moveConnections();
//}

void WtFlowScene::removeNode(WtNode& node)
{
	// call signal
	//nodeDeleted(node);
	clearNode(node);
}

void WtFlowScene::clearNode(WtNode& node)
{
	for (auto portType : { PortType::In, PortType::Out })
	{
		auto nodeState = node.nodeState();
		auto const& nodeEntries = nodeState.getEntries(portType);

		for (auto& connections : nodeEntries)
		{
			for (auto const& pair : connections)
			{
				deleteConnection(*pair.second);
			}
		}
	}
	_nodes.erase(node.id());
}

std::unordered_map<Wt::Guid, std::unique_ptr<WtNode> > const& WtFlowScene::nodes() const
{
	return _nodes;
}

std::unordered_map<Wt::Guid, std::shared_ptr<WtConnection> > const& WtFlowScene::connections() const
{
	return _connections;
}

void WtFlowScene::clearScene()
{
	//Manual node cleanup. Simply clearing the holding datastructures doesn't work, the code crashes when
// there are both nodes and connections in the scene. (The data propagation internal logic tries to propagate
// data through already freed connections.)
	while (_connections.size() > 0)
	{
		deleteConnection(*_connections.begin()->second);
	}

	while (_nodes.size() > 0)
	{
		clearNode(*_nodes.begin()->second);
	}
}

std::vector<WtNode*> WtFlowScene::allNodes() const
{
	std::vector<WtNode*> nodes;

	std::transform(_nodes.begin(),
		_nodes.end(),
		std::back_inserter(nodes),
		[](std::pair<Wt::Guid const, std::unique_ptr<WtNode>> const& p) { return p.second.get(); });

	return nodes;
}

//std::vector<WtNode*> WtFlowScene::selectedNodes() const
//{
//	//QList<QGraphicsItem*> graphicsItems = selectedItems();
//
//	//std::vector<WtNode*> ret;
//	//ret.reserve(graphicsItems.size());
//
//	//for (QGraphicsItem* item : graphicsItems)
//	//{
//	//	auto ngo = qgraphicsitem_cast<QtNodeGraphicsObject*>(item);
//
//	//	if (ngo != nullptr)
//	//	{
//	//		ret.push_back(&ngo->node());
//	//	}
//	//}
//
//	//return ret;
//}