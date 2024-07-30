#include "WtNodeFlowScene.h"

WtNodeFlowScene::WtNodeFlowScene(Wt::WPainter* painter)
{
	auto classMap = dyno::Object::getClassMap();
	auto ret = std::make_shared<WtDataModelRegistry>();
	int id = 0;
	for (auto const c : *classMap)
	{
		id++;

		std::string str = c.first;
		auto obj = dyno::Object::createObject(str);
		std::shared_ptr<dyno::Node> node(dynamic_cast<dyno::Node*>(obj));

		if (node != nullptr)
		{
			WtDataModelRegistry::RegistryItemCreator creator = [str]()
				{
					auto node_obj = dyno::Object::createObject(str);
					std::shared_ptr<dyno::Node> new_node(dynamic_cast<dyno::Node*>(node_obj));
					auto dat = std::make_unique<WtNodeWidget>(std::move(new_node));
					return dat;
				};

			std::string category = node->getNodeType();
			ret->registerModel<WtNodeWidget>(category, creator);
		}
	}

	//this->setRegistry(ret);
	//createNodeGraphView();
	//reorderAllNodes();
	//connect(this, &QtFlowScene::nodeMoved, this, &QtNodeFlowScene::moveNode);
	//connect(this, &QtFlowScene::nodePlaced, this, &QtNodeFlowScene::addNode);
	//connect(this, &QtFlowScene::nodeDeleted, this, &QtNodeFlowScene::deleteNode);
	//connect(this, &QtFlowScene::nodeHotKey0Checked, this, &QtNodeFlowScene::enableRendering);
	//connect(this, &QtFlowScene::nodeHotKey1Checked, this, &QtNodeFlowScene::enablePhysics);
	////connect(this, &QtFlowScene::nodeHotKey2Checked, this, &QtNodeFlowScene::Key2_Signal);
	//connect(this, &QtFlowScene::nodeContextMenu, this, &QtNodeFlowScene::showContextMenu);
}

WtNodeFlowScene::~WtNodeFlowScene() {}

void WtNodeFlowScene::createNodeGraphView()
{
	auto scn = dyno::SceneGraphFactory::instance()->active();

	std::map<dyno::ObjectId, WtNode*> nodeMap;

	auto addNodeWidget = [&](std::shared_ptr<Node> m) -> void
		{
			auto mId = m->objectId();

			auto type = std::make_unique<WtNodeWidget>(m);

			auto& node = this->createNode(std::move(type));

			nodeMap[mId] = &node;

			Wt::WPointF posView(m->bx(), m->by());

			//TODO:Position
			/*node.nodeGraphicsObject().setPos(posView);
			node.nodeGraphicsObject().setHotKey0Checked(m->isVisible());
			node.nodeGraphicsObject().setHotKey1Checked(m->isActive());*/
		};
}