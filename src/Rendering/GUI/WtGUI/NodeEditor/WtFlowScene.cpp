#include "WtFlowScene.h"

WtFlowScene::WtFlowScene() {}

WtFlowScene::WtFlowScene(std::shared_ptr<WtDataModelRegistry> registry, Wt::WPainter* painter)
	: _registry(std::move(registry))
{
}

WtFlowScene::~WtFlowScene()
{
}

WtNode& WtFlowScene::createNode(std::unique_ptr<WtNodeDataModel>&& dataModel)
{
	auto node = detail::make_unique<WtNode>(std::move(dataModel));
	auto ngo = detail::make_unique<WtNodeGraphicsObject>(*this, *node);

	node->setGraphicsObject(std::move(ngo));

	auto nodePtr = node.get();
	_nodes[node->id()] = std::move(node);

	//signal for node created
	//nodeCreated(*nodePtr);

	return *nodePtr;
}