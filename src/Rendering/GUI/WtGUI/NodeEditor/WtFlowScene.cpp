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

//std::shared_ptr<WtConnection> createConnection(PortType connectedPort,
//	WtNode& node,
//	PortIndex portIndex)
//{
//	auto connection = std::make_shared<WtConnection>(connectedPort, node, portIndex);
//
//	//auto cgo = detail::make_unique<WtConnectionGraphicsObject>(*this, *connection);
//}
//
//std::shared_ptr<WtConnection> createConnection(WtNode& nodeIn,
//	PortIndex portIndexIn,
//	WtNode& nodeOut,
//	PortIndex portIndexOut,
//	TypeConverter const& converter = TypeConverter{})
//{
//}

WtNode& WtFlowScene::createNode(std::unique_ptr<WtNodeDataModel>&& dataModel)
{
	auto node = detail::make_unique<WtNode>(std::move(dataModel));
	auto ngo = detail::make_unique<WtNodeGraphicsObject>(*this, *node);

	node->setGraphicsObject(std::move(ngo));

	auto nodePtr = node.get();
	_nodes[node->id()] = std::move(node);

	return *nodePtr;
}

//QtNode&
//	QtFlowScene::
//	restoreNode(QJsonObject const& nodeJson)
//{
//	QString modelName = nodeJson["model"].toObject()["name"].toString();

//	auto dataModel = registry().create(modelName);

//	if (!dataModel)
//		throw std::logic_error(std::string("No registered model with name ") +
//			modelName.toLocal8Bit().data());

//	auto node = detail::make_unique<QtNode>(std::move(dataModel));
//	auto ngo = detail::make_unique<QtNodeGraphicsObject>(*this, *node);
//	node->setGraphicsObject(std::move(ngo));

//	node->restore(nodeJson);

//	auto nodePtr = node.get();
//	_nodes[node->id()] = std::move(node);

//	nodePlaced(*nodePtr);
//	nodeCreated(*nodePtr);
//	return *nodePtr;
//}

WtDataModelRegistry& WtFlowScene::registry() const
{
	return *_registry;
}

void WtFlowScene::setRegistry(std::shared_ptr<WtDataModelRegistry> registry)
{
	_registry = std::move(registry);
}

void WtFlowScene::removeNode(WtNode& node)
{
	//nodeDeleted(node);
	clearNode(node);
}

void WtFlowScene::clearNode(WtNode& node)
{
	//for (auto portType : { PortType::In, PortType::Out })
	//{
	//	auto nodeState = node.nodeState();
	//	auto const& nodeEntries = nodeState.getEntries(portType);

	//	for (auto& connections : nodeEntries)
	//	{
	//		for (auto const& pair : connections)
	//			deleteConnection(*pair.second);
	//	}
	//}
	//_nodes.erase(node.id());
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