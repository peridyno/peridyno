#include "WtNodeFlowScene.h"

WtNodeFlowScene::WtNodeFlowScene(Wt::WPainter* painter)
	: WtFlowScene()
{
	_painter = painter;

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

	this->setRegistry(ret);

	createNodeGraphView();
	reorderAllNodes();

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

			auto& node = this->createNode(std::move(type), _painter);

			nodeMap[mId] = &node;

			Wt::WPointF posView(m->bx(), m->by());

			node.nodeGraphicsObject().setPos(posView);
			std::cout << "!!!" << std::endl;
			std::cout << m->bx() << std::endl;
			std::cout <<  m->by() << std::endl;
			node.nodeGraphicsObject().setHotKey0Checked(m->isVisible());
			node.nodeGraphicsObject().setHotKey1Checked(m->isActive());
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
								createConnection(*inBlock, i, *outBlock, 0, _painter);
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
									createConnection(*inBlock, i, *outBlock, 0, _painter);
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
									createConnection(*inBlock, i + ports.size(), *outBlock, outFieldIndex, _painter);
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

	//clearScene();
	//for (auto it = scn->begin(); it != scn->end(); it++)
	//{
	//	auto node_ptr = it.get();
	//	std::cout << node_ptr->getClassInfo()->getClassName() << ": " << node_ptr.use_count() << std::endl;
	//}

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

void WtNodeFlowScene::moveNode(WtNode& n, const Wt::WPointF& newLocaton)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr)
	{
		auto node = nodeData->getNode();
		node->setBlockCoord(newLocaton.x(), newLocaton.y());
	}
}

void WtNodeFlowScene::addNode(WtNode& n)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr)
	{
		auto scn = dyno::SceneGraphFactory::instance()->active();
		scn->addNode(nodeData->getNode());
	}
}

void WtNodeFlowScene::addNodeByString(std::string NodeName)
{
	Wt::log("info") << NodeName;

	auto node_obj = dyno::Object::createObject(NodeName);
	std::shared_ptr<dyno::Node> new_node(dynamic_cast<dyno::Node*>(node_obj));
	auto dat = std::make_unique<WtNodeWidget>(std::move(new_node));

	if (dat != nullptr)
	{
		auto scn = dyno::SceneGraphFactory::instance()->active();
		scn->addNode(dat->getNode());
	}
	else
	{
		Wt::log("info") << "nullptr";
	}

	int mId;

	auto addNodeWidget = [&](std::shared_ptr<Node> m) -> void
		{
			mId = m->objectId();

			auto type = std::make_unique<WtNodeWidget>(m);

			auto& node = this->createNode(std::move(type), _painter);

			Wt::WPointF posView(m->bx(), m->by());

			node.nodeGraphicsObject().setPos(posView);

			// signal
			//this->nodePlaced(node);
		};

	auto scn = dyno::SceneGraphFactory::instance()->active();

	int x = 0;

	for (auto it = scn->begin(); it != scn->end(); it++)
	{
		if (x == mId)
		{
			addNodeWidget(it.get());
			break;
		}
		x++;
	}
	addNodeWidget(dat->getNode());
}

void WtNodeFlowScene::deleteNode(WtNode& n)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr)
	{
		auto scn = dyno::SceneGraphFactory::instance()->active();
		scn->deleteNode(nodeData->getNode());
	}
}

void WtNodeFlowScene::createWtNode(std::shared_ptr<dyno::Node> node)
{
	if (node == nullptr)
		return;

	auto qNodeWidget = std::make_unique<WtNodeWidget>(node);
	auto& qNode = createNode(std::move(qNodeWidget), _painter);

	//Calculate the position for the newly create node to avoid overlapping
	auto& _nodes = this->nodes();
	float y = -10000.0f;
	for (auto const& _node : _nodes)
	{
		WtNodeGeometry& geo = _node.second->nodeGeometry();
		WtNodeGraphicsObject& obj = _node.second->nodeGraphicsObject();

		float h = geo.height();

		Wt::WPointF pos = obj.pos();

		y = std::max(y, float(pos.y() + h));
	}

	Wt::WPointF posView(0.0f, y + 50.0f);

	qNode.nodeGraphicsObject().setPos(posView);

	// signal
	// emit nodePlaced(qNode);
}

void WtNodeFlowScene::enableRendering(WtNode& n, bool checked)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr) {
		auto node = nodeData->getNode();
		node->setVisible(checked);
	}
}

void WtNodeFlowScene::enablePhysics(WtNode& n, bool checked)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr) {
		auto node = nodeData->getNode();
		node->setActive(checked);
	}
}

void WtNodeFlowScene::showContextMenu(WtNode& n, const Wt::WPointF& pos)
{
}

void WtNodeFlowScene::showThisNodeOnly(WtNode& n)
{
	auto nodes = this->allNodes();
	for (auto node : nodes)
	{
		if (node->id() == n.id())
		{
			node->nodeGraphicsObject().setHotKey1Hovered(true);
			this->enableRendering(*node, true);
		}
		else
		{
			node->nodeGraphicsObject().setHotKey1Hovered(false);
			this->enableRendering(*node, false);
		}
	}

	this->updateNodeGraphView();

	nodes.clear();
}

void WtNodeFlowScene::showAllNodes()
{
	auto nodes = this->allNodes();
	for (auto node : nodes)
	{
		this->enableRendering(*node, true);
	}

	this->updateNodeGraphView();

	nodes.clear();
}

void WtNodeFlowScene::activateThisNodeOnly(WtNode& n)
{
	auto nodes = this->allNodes();
	for (auto node : nodes)
	{
		if (node->id() == n.id())
		{
			node->nodeGraphicsObject().setHotKey0Hovered(true);
			this->enablePhysics(*node, true);
		}
		else
		{
			node->nodeGraphicsObject().setHotKey0Hovered(false);
			this->enablePhysics(*node, false);
		}
	}

	this->updateNodeGraphView();

	nodes.clear();
}

void WtNodeFlowScene::activateAllNodes()
{
	auto nodes = this->allNodes();
	for (auto node : nodes)
	{
		auto a = node->nodeDataModel();
	}

	this->updateNodeGraphView();

	nodes.clear();
}

void WtNodeFlowScene::autoSyncAllNodes(bool autoSync)
{
	auto nodes = this->allNodes();
	for (auto node : nodes)
	{
		auto dataModel = dynamic_cast<WtNodeWidget*>(node->nodeDataModel());
		if (dataModel != nullptr)
		{
			auto dNode = dataModel->getNode();
			dNode->setAutoSync(autoSync);
		}
	}

	nodes.clear();
}

void WtNodeFlowScene::autoSyncAllDescendants(WtNode& n, bool autoSync)
{
	auto scn = dyno::SceneGraphFactory::instance()->active();

	class ToggleAutoSyncAct : public dyno::Action
	{
	public:
		ToggleAutoSyncAct(bool autoSync) { mAutoSync = autoSync; }

		void process(Node* node) override {
			node->setAutoSync(mAutoSync);
		}

	private:
		bool mAutoSync = true;
	};

	ToggleAutoSyncAct act(autoSync);

	auto dataModel = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());
	if (dataModel != nullptr)
	{
		auto dNode = dataModel->getNode();
		if (dNode == nullptr) return;

		scn->traverseForward(dNode, &act);
	}
}

//TODO: show a message on how to use this node
void WtNodeFlowScene::showHelper(WtNode& n)
{
	Wt::log("info") << "Show something about this node";
}

void WtNodeFlowScene::reorderAllNodes()
{
	auto scn = dyno::SceneGraphFactory::instance()->active();

	dyno::DirectedAcyclicGraph graph;

	auto constructDAG = [&](std::shared_ptr<Node> nd) -> void
		{
			auto inId = nd->objectId();
			auto ports = nd->getImportNodes();

			graph.addOtherVertices(inId);
			graph.removeID();

			bool NodeConnection = false;
			bool FieldConnection = false;

			for (int i = 0; i < ports.size(); i++)
			{
				dyno::NodePortType pType = ports[i]->getPortType();

				if (dyno::Single == pType)
				{
					auto node = ports[i]->getNodes()[0];
					if (node != nullptr)
					{
						auto outId = node->objectId();

						graph.addEdge(outId, inId);

						graph.removeID(outId, inId);
					}
				}
				else if (dyno::Multiple == pType)
				{
					auto& nodes = ports[i]->getNodes();

					for (int j = 0; j < nodes.size(); j++)
					{
						if (nodes[j] != nullptr)
						{
							auto outId = nodes[j]->objectId();

							graph.addEdge(outId, inId);
							graph.removeID(outId, inId);
						}
					}
					//nodes.clear();
				}
			}

			auto fieldInp = nd->getInputFields();

			for (int i = 0; i < fieldInp.size(); i++)//遍历每个Node的Inputfield
			{
				auto fieldSrc = fieldInp[i]->getSource();

				if (fieldSrc != nullptr) {
					auto parSrc = fieldSrc->parent();

					if (parSrc != nullptr)
					{
						Node* nodeSrc = dynamic_cast<Node*>(parSrc);

						// Otherwise parSrc is a field of Module
						if (nodeSrc == nullptr)
						{
							dyno::Module* moduleSrc = dynamic_cast<dyno::Module*>(parSrc);
							if (moduleSrc != nullptr)
								nodeSrc = moduleSrc->getParentNode();
						}

						if (nodeSrc != nullptr)
						{
							auto outId = nodeSrc->objectId();

							graph.addEdge(outId, inId);

							graph.removeID(outId, inId);
						}
					}
				}
			}
		};

	for (auto it = scn->begin(); it != scn->end(); it++)
	{
		constructDAG(it.get());
	}

	dyno::AutoLayoutDAG layout(&graph);
	layout.update();

	//Set up the mapping from ObjectId to QtNode
	auto& _nodes = this->nodes();
	std::map<dyno::ObjectId, WtNode*> qtNodeMapper;
	std::map<dyno::ObjectId, Node*> nodeMapper;

	for (auto const& _node : _nodes)
	{
		auto const& qtNode = _node.second;
		auto model = qtNode->nodeDataModel();

		auto nodeData = dynamic_cast<WtNodeWidget*>(model);

		if (model != nullptr)
		{
			auto node = nodeData->getNode();
			if (node != nullptr)
			{
				qtNodeMapper[node->objectId()] = qtNode.get();
				nodeMapper[node->objectId()] = node.get();
			}
		}
	}

	float tempOffsetY = 0.0f;
	float offsetX = 0.0f;

	for (size_t l = 0; l < layout.layerNumber(); l++)
	{
		auto& xc = layout.layer(l);

		float offsetY = 0.0f;
		float xMax = 0.0f;

		for (size_t index = 0; index < xc.size(); index++)
		{
			dyno::ObjectId id = xc[index];
			if (qtNodeMapper.find(id) != qtNodeMapper.end())
			{
				WtNode* qtNode = qtNodeMapper[id];
				WtNodeGeometry& geo = qtNode->nodeGeometry();

				float w = geo.width();
				float h = geo.height();

				xMax = std::max(xMax, w);

				Node* node = nodeMapper[id];

				node->setBlockCoord(offsetX, offsetY);

				offsetY += (h + mDy);
			}
		}

		offsetX += (xMax + mDx);

		tempOffsetY = std::max(tempOffsetY, offsetY);
	}

	//离散节点的排序
	auto otherVertices = layout.getOtherVertices();
	float width = 0;
	float heigth = 0;
	std::set<dyno::ObjectId>::iterator it;

	float ofstY = tempOffsetY;
	float ofstX = 0;

	for (it = otherVertices.begin(); it != otherVertices.end(); it++)
	{
		dyno::ObjectId id = *it;

		if (qtNodeMapper.find(id) != qtNodeMapper.end())
		{
			WtNode* qtNode = qtNodeMapper[id];
			WtNodeGeometry& geo = qtNode->nodeGeometry();
			width = geo.width();
			heigth = geo.height();

			Node* node = nodeMapper[id];

			node->setBlockCoord(ofstX, ofstY);
			ofstX += width + mDx;
		}
	}

	qtNodeMapper.clear();
	nodeMapper.clear();

	updateNodeGraphView();
}