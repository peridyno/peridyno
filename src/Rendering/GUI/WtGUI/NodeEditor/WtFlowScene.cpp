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

WtFlowScene::~WtFlowScene() {}

WtNode& WtFlowScene::createNode(std::unique_ptr<WtNodeDataModel>&& dataModel)
{
	auto node = detail::make_unique<WtNode>(std::move(dataModel));
	auto ngo = detail::make_unique<WtNodeGraphicsObject>(*this, *node);

	node->setGraphicsObject(std::move(ngo));

	auto nodePtr = node.get();
	_nodes[node->id()] = std::move(node);

	return *nodePtr;
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
