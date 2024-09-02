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

WtNode& WtFlowScene::createNode(std::unique_ptr<WtNodeDataModel>&& dataModel, Wt::WPainter* painter)
{
	auto node = detail::make_unique<WtNode>(std::move(dataModel));
	auto ngo = detail::make_unique<WtNodeGraphicsObject>(*this, *node, painter);

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
	std::set<boost::uuids::uuid> visitedNodesSet;

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
					if (visitedNodesSet.find(conn.second->getNode(PortType::Out)->id()) == visitedNodesSet.end())
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
	//return node.nodeGraphicsObject().pos();
}

Wt::WPointF WtFlowScene::setNodePosition(WtNode& node, const Wt::WPointF& pos) const
{
	node.nodeGraphicsObject().setPos(pos);
	node.nodeGraphicsObject().moveConnections();
}

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

void WtFlowScene::clearScene()
{
}

std::vector<WtNode*> WtFlowScene::allNodes() const
{
	std::vector<WtNode*> nodes;

	std::transform(_nodes.begin(),
		_nodes.end(),
		std::back_inserter(nodes),
		[](std::pair<boost::uuids::uuid const, std::unique_ptr<WtNode>> const& p) { return p.second.get(); });

	return nodes;
}