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

	//auto root = scn->getRootNode();

	//SceneGraph::Iterator it_end(nullptr);

	auto addNodeWidget = [&](std::shared_ptr<Node> m) -> void
		{
			auto mId = m->objectId();

			auto type = std::make_unique<WtNodeWidget>(m);

			auto& node = this->createNode(std::move(type));

			nodeMap[mId] = &node;

			Wt::WPointF posView(m->bx(), m->by());

			//node.nodeGraphicsObject().setPos(posView);
			node.nodeGraphicsObject().setHotKey0Checked(m->isVisible());
			node.nodeGraphicsObject().setHotKey1Checked(m->isActive());

			//singal
			//this->nodePlaced(node);
		};

	for (auto it = scn->begin(); it != scn->end(); it++)
	{
		addNodeWidget(it.get());
	}

	auto createNodeConnections = [&](std::shared_ptr<Node> nd) -> void
		{
			auto inId = nd->objectId();

			if (nodeMap.find(inId) != nodeMap.end())
			{
				auto inBlock = nodeMap[nd->objectId()];

				auto ports = nd->getImportNodes();

				for (int i = 0; i < ports.size(); i++)
				{
					dyno::NodePortType pType = ports[i]->getPortType();
					if (dyno::Single == pType)
					{
						auto node = ports[i]->getNodes()[0];
						if (node != nullptr)
						{
							auto outId = node->objectId();
							if (nodeMap.find(outId) != nodeMap.end())
							{
								auto outBlock = nodeMap[node->objectId()];
								//createConnection(*inBlock, i, *outBlock, 0);
							}
						}
					}
					else if (dyno::Multiple == pType)
					{
						//TODO: a weird problem exist here, if the expression "auto& nodes = ports[i]->getNodes()" is used,
						//we still have to call clear to avoid memory leak.
						auto& nodes = ports[i]->getNodes();
						//ports[i]->clear();
						for (int j = 0; j < nodes.size(); j++)
						{
							if (nodes[j] != nullptr)
							{
								auto outId = nodes[j]->objectId();
								if (nodeMap.find(outId) != nodeMap.end())
								{
									auto outBlock = nodeMap[outId];
									//createConnection(*inBlock, i, *outBlock, 0);
								}
							}
						}
						//nodes.clear();
					}
				}

				auto fieldInp = nd->getInputFields();
				for (int i = 0; i < fieldInp.size(); i++)
				{
					auto fieldSrc = fieldInp[i]->getSource();
					if (fieldSrc != nullptr) {
						auto parSrc = fieldSrc->parent();
						if (parSrc != nullptr)
						{
							//To handle fields from node states or outputs
							dyno::Node* nodeSrc = dynamic_cast<dyno::Node*>(parSrc);

							//To handle fields that are exported from module outputs
							if (nodeSrc == nullptr)
							{
								dyno::Module* moduleSrc = dynamic_cast<dyno::Module*>(parSrc);
								if (moduleSrc != nullptr)
									nodeSrc = moduleSrc->getParentNode();
							}

							if (nodeSrc != nullptr)
							{
								auto outId = nodeSrc->objectId();
								auto fieldsOut = nodeSrc->getOutputFields();

								unsigned int outFieldIndex = 0;
								bool fieldFound = false;
								for (auto f : fieldsOut)
								{
									if (f == fieldSrc)
									{
										fieldFound = true;
										break;
									}
									outFieldIndex++;
								}

								if (nodeMap[outId]->nodeDataModel()->allowExported()) outFieldIndex++;

								if (fieldFound && nodeMap.find(outId) != nodeMap.end())
								{
									auto outBlock = nodeMap[outId];
									//createConnection(*inBlock, i + ports.size(), *outBlock, outFieldIndex);
								}
							}
						}
					}
				}
			}
		};

	for (auto it = scn->begin(); it != scn->end(); it++)
	{
		createNodeConnections(it.get());
	}

	// 		// 	clearScene();
	// 		// 
	// 		for (auto it = scn->begin(); it != scn->end(); it++)
	// 		{
	// 			auto node_ptr = it.get();
	// 			std::cout << node_ptr->getClassInfo()->getClassName() << ": " << node_ptr.use_count() << std::endl;
	// 		}

	nodeMap.clear();
}

void WtNodeFlowScene::updateNodeGraphView()
{
	disableEditing();

	clearScene();

	createNodeGraphView();

	enableEditing();
}

void WtNodeFlowScene::fieldUpdated(dyno::FBase* field, int status)
{
	disableEditing();

	clearScene();

	//auto f = status == Qt::Checked ? field->promoteOuput() : field->demoteOuput();

	createNodeGraphView();

	enableEditing();
}

void WtNodeFlowScene::enableEditing()
{
	mEditingEnabled = true;

	auto allNodes = this->allNodes();

	for (auto node : allNodes)
	{
		auto model = dynamic_cast<WtNodeWidget*>(node->nodeDataModel());
		if (model != nullptr)
		{
			model->enableEditing();
		}
	}
}

void WtNodeFlowScene::disableEditing()
{
	mEditingEnabled = false;

	auto allNodes = this->allNodes();

	for (auto node : allNodes)
	{
		auto model = dynamic_cast<WtNodeWidget*>(node->nodeDataModel());
		if (model != nullptr)
		{
			model->disableEditing();
		}
	}
}