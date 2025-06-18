#include "QtNodeFlowScene.h"
#include "QtNodeWidget.h"

#include "nodes/QNode"

#include "Format.h"

#include "Object.h"
#include "NodeIterator.h"
#include "NodePort.h"
#include "Action.h"
#include "DirectedAcyclicGraph.h"
#include "AutoLayoutDAG.h"
#include "SceneGraphFactory.h"

#include <QtWidgets/QMessageBox>
#include <QKeySequence>
#include <QShortcut>
#include <QMenu>
#include "iostream"

namespace Qt
{
	QtNodeFlowScene::QtNodeFlowScene(std::shared_ptr<QtDataModelRegistry> registry, QObject* parent)
		: QtFlowScene(registry, parent)
	{
	}

	QtNodeFlowScene::QtNodeFlowScene(QObject* parent)
		: QtFlowScene(parent)
	{
		auto classMap = dyno::Object::getClassMap();

		auto ret = std::make_shared<QtDataModelRegistry>();
		int id = 0;
		for (auto const c : *classMap)
		{
			id++;

			QString str = QString::fromStdString(c.first);
			auto obj = dyno::Object::createObject(str.toStdString());
			std::shared_ptr<dyno::Node> node(dynamic_cast<dyno::Node*>(obj));

			if (node != nullptr)
			{
				QtDataModelRegistry::RegistryItemCreator creator = [str]() {
					auto node_obj = dyno::Object::createObject(str.toStdString());
					std::shared_ptr<dyno::Node> new_node(dynamic_cast<dyno::Node*>(node_obj));
					auto dat = std::make_unique<QtNodeWidget>(std::move(new_node));
					return dat;
					};

				QString category = dyno::FormatBlockCaptionName(node->getNodeType());
				ret->registerModel<QtNodeWidget>(category, creator);
			}
		}

		this->setRegistry(ret);

		createNodeGraphView();
		reorderAllNodes();

		connect(this, &QtFlowScene::nodeMoved, this, &QtNodeFlowScene::moveNode);
		connect(this, &QtFlowScene::nodePlaced, this, &QtNodeFlowScene::addNode);
		connect(this, &QtFlowScene::nodeDeleted, this, &QtNodeFlowScene::deleteNode);

// 		connect(this, &QtFlowScene::nodeHotKey0Checked, this, &QtNodeFlowScene::enableRendering);
// 		connect(this, &QtFlowScene::nodeHotKey1Checked, this, &QtNodeFlowScene::enablePhysics);
		
		connect(this, &QtFlowScene::nodeHotKeyClicked, [this](QtNode& n, bool checked, int buttonId) {

			switch (buttonId) 
			{
				case 0:
					enableRendering(n, checked);
					break;
				case 1:
					enablePhysics(n, checked);
					break;
				case 2:
					enableAutoSync(n, checked);
					break;
				case 3:
					resetNode(n);
					break;

				default:
					break;
			}
				
			});
		//connect(this, &QtFlowScene::nodeHotKey2Checked, this, &QtNodeFlowScene::Key2_Signal);

		connect(this, &QtFlowScene::nodeContextMenu, this, &QtNodeFlowScene::showContextMenu);
	}

	QtNodeFlowScene::~QtNodeFlowScene()
	{
		clearScene();
	}

	void QtNodeFlowScene::createNodeGraphView()
	{
		auto scn = dyno::SceneGraphFactory::instance()->active();

		std::map<dyno::ObjectId, QtNode*> nodeMap;

		//auto root = scn->getRootNode();

		//SceneGraph::Iterator it_end(nullptr);

		auto addNodeWidget = [&](std::shared_ptr<Node> m) -> void
			{
				auto mId = m->objectId();

				auto type = std::make_unique<QtNodeWidget>(m);

				auto& node = this->createNode(std::move(type));

				nodeMap[mId] = &node;

				QPointF posView(m->bx(), m->by());

			node.nodeGraphicsObject().setPos(posView);

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
									createConnection(*inBlock, i, *outBlock, 0);
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
										createConnection(*inBlock, i, *outBlock, 0);
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

									uint outFieldIndex = 0;
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
										createConnection(*inBlock, i + ports.size(), *outBlock, outFieldIndex);
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

	void QtNodeFlowScene::updateNodeGraphView()
	{
		disableEditing();

		clearScene();

		createNodeGraphView();

		enableEditing();
	}

	void QtNodeFlowScene::fieldUpdated(dyno::FBase* field, int status)
	{
		disableEditing();

		clearScene();

		auto f = status == Qt::Checked ? field->promoteOuput() : field->demoteOuput();

		createNodeGraphView();

		enableEditing();
	}

	void QtNodeFlowScene::moveNode(QtNode& n, const QPointF& newLocation)
	{
		auto nodeData = dynamic_cast<QtNodeWidget*>(n.nodeDataModel());

		if (mEditingEnabled && nodeData != nullptr) {
			auto node = nodeData->getNode();
			node->setBlockCoord(newLocation.x(), newLocation.y());
		}
	}

	void QtNodeFlowScene::addNode(QtNode& n)
	{
		auto nodeData = dynamic_cast<QtNodeWidget*>(n.nodeDataModel());

		if (mEditingEnabled && nodeData != nullptr) {
			auto scn = dyno::SceneGraphFactory::instance()->active();
			scn->addNode(nodeData->getNode());
		}
	}

	void QtNodeFlowScene::addNodeByString(std::string NodeName) {
		std::cout << NodeName << std::endl;

		auto node_obj = dyno::Object::createObject(NodeName);
		std::shared_ptr<dyno::Node> new_node(dynamic_cast<dyno::Node*>(node_obj));
		auto dat = std::make_unique<QtNodeWidget>(std::move(new_node));

		if (dat != nullptr) {
			auto scn = dyno::SceneGraphFactory::instance()->active();
			scn->addNode(dat->getNode());
		}
		else {
			std::cout << "nullptr" << std::endl;
		}
		int mId;
		auto addNodeWidget = [&](std::shared_ptr<Node> m) -> void
			{
				mId = m->objectId();

				auto type = std::make_unique<QtNodeWidget>(m);

				auto& node = this->createNode(std::move(type));

				QPointF posView(m->bx(), m->by());

				node.nodeGraphicsObject().setPos(posView);

				this->nodePlaced(node);
			};
		auto scn = dyno::SceneGraphFactory::instance()->active();
		int x = 0;
		for (auto it = scn->begin(); it != scn->end(); it++)
		{
			if (x == mId) {
				addNodeWidget(it.get());
				break;
			}
			x++;
		}
		addNodeWidget(dat->getNode());
	}

	void QtNodeFlowScene::enableEditing()
	{
		mEditingEnabled = true;

		auto allNodes = this->allNodes();

		for (auto node : allNodes)
		{
			auto model = dynamic_cast<QtNodeWidget*>(node->nodeDataModel());
			if (model != nullptr)
			{
				model->enableEditing();
			}
		}
	}

	void QtNodeFlowScene::disableEditing()
	{
		mEditingEnabled = false;

		auto allNodes = this->allNodes();

		for (auto node : allNodes)
		{
			auto model = dynamic_cast<QtNodeWidget*>(node->nodeDataModel());
			if (model != nullptr)
			{
				model->disableEditing();
			}
		}
	}

	void QtNodeFlowScene::deleteNode(QtNode& n)
	{
		auto nodeData = dynamic_cast<QtNodeWidget*>(n.nodeDataModel());

		if (mEditingEnabled && nodeData != nullptr) {
			auto scn = dyno::SceneGraphFactory::instance()->active();
			scn->deleteNode(nodeData->getNode());

			emit this->nodeDeselected();
		}
	}

	void QtNodeFlowScene::createQtNode(std::shared_ptr<dyno::Node> node)
	{
		if (node == nullptr)
			return;

		auto qNodeWdiget = std::make_unique<QtNodeWidget>(node);
		auto& qNode = createNode(std::move(qNodeWdiget));

		//Calculate the position for the newly create node to avoid overlapping
		auto& _nodes = this->nodes();
		float y = -10000.0f;
		for (auto const& _node : _nodes)
		{
			NodeGeometry& geo = _node.second->nodeGeometry();
			QtNodeGraphicsObject& obj = _node.second->nodeGraphicsObject();

			float h = geo.height();

			QPointF pos = obj.pos();

			y = std::max(y, float(pos.y() + h));
		}

		QPointF posView(0.0f, y + 50.0f);

		qNode.nodeGraphicsObject().setPos(posView);

		emit nodePlaced(qNode);
	}

	void QtNodeFlowScene::enableRendering(QtNode& n, bool checked)
	{
		auto nodeData = dynamic_cast<QtNodeWidget*>(n.nodeDataModel());

		if (mEditingEnabled && nodeData != nullptr) {
			auto node = nodeData->getNode();
			node->setVisible(!checked);
			node->graphicsPipeline()->enable();
			node->graphicsPipeline()->update();
		}
	}

	void QtNodeFlowScene::enableAutoSync(QtNode& n, bool checked) 
	{
		auto nodeData = dynamic_cast<QtNodeWidget*>(n.nodeDataModel());

		if (mEditingEnabled && nodeData != nullptr) {
			auto node = nodeData->getNode();
			node->setAutoSync(checked);
		}
	}

	void QtNodeFlowScene::enablePhysics(QtNode& n, bool checked)
	{
		auto nodeData = dynamic_cast<QtNodeWidget*>(n.nodeDataModel());

		if (mEditingEnabled && nodeData != nullptr) {
			auto node = nodeData->getNode();
			node->setActive(!checked);
			node->animationPipeline()->update();
		}
	}

	void QtNodeFlowScene::resetNode(QtNode& n) 
	{
		auto nodeData = dynamic_cast<QtNodeWidget*>(n.nodeDataModel());

		if (mEditingEnabled && nodeData != nullptr) {
			auto node = nodeData->getNode();
			node->reset();
		}
	}

	void QtNodeFlowScene::showContextMenu(QtNode& n, const QPointF& pos)
	{
		
		auto qDataModel = dynamic_cast<QtNodeWidget*>(n.nodeDataModel());
		auto node = qDataModel->getNode();
		if (node == nullptr) {
			return;
		}

		auto menu = new QMenu;
		menu->setStyleSheet("QMenu{color:white;border: 1px solid black;} "); //QMenu::item:selected {background-color : #4b586a;}

		auto openAct = new QAction("Open", this);

		auto showAllNodesAct = new QAction("Show All Nodes", this);
		auto showThisNodeOnlyAct = new QAction("Show This Only", this);

		showAllNodesAct->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_V));
		showThisNodeOnlyAct->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_H));

		auto delAct = new QAction("Delete", this);
		auto helpAct = new QAction("Help", this);

		menu->addAction(openAct);

		menu->addSeparator();
		menu->addAction(showThisNodeOnlyAct);
		menu->addAction(showAllNodesAct);


		auto resetNodeAct = new QAction("Reset This Node", this);
		auto activateThisNodeOnlyAct = new QAction("Activate This Only", this);
		auto activateAllNodesAct = new QAction("Activate All Nodes", this);
		menu->addSeparator();
		menu->addAction(resetNodeAct);

		menu->addSeparator();
		menu->addAction(activateThisNodeOnlyAct);
		menu->addAction(activateAllNodesAct);

		menu->addSeparator();
		auto autoSyncAct = new QAction("Auto-Sync", this);
		autoSyncAct->setCheckable(true);
		autoSyncAct->setChecked(node->isAutoSync());

		auto enableDiscendants = new QAction("Enable Descendants' Auto-Sync(s)", this);
		auto disableDiscendants = new QAction("Disable Descendants' Auto-Sync(s)", this);
		auto enableAutoSync = new QAction("Enable All Auto-Sync(s)", this);
		auto disableAutoSync = new QAction("Disable All Auto-Sync(s)", this);
		menu->addAction(autoSyncAct);
		menu->addAction(enableDiscendants);
		menu->addAction(disableDiscendants);
		menu->addAction(enableAutoSync);
		menu->addAction(disableAutoSync);

		menu->addSeparator();
		menu->addAction(delAct);

		menu->addSeparator();
		menu->addAction(helpAct);

		connect(openAct, &QAction::triggered, this, [&]() { nodeDoubleClicked(n); });


		connect(showAllNodesAct, &QAction::triggered, this, [&]() {
			showAllNodes();
			});

		connect(showThisNodeOnlyAct, &QAction::triggered, this, [&]() {
			showThisNodeOnly(n);
			});

		connect(resetNodeAct, &QAction::triggered, this, [&]() {
			resetNode(n);
			});

		connect(activateAllNodesAct, &QAction::triggered, this, [&]() {
			activateAllNodes();
			});

		connect(activateThisNodeOnlyAct, &QAction::triggered, this, [&]() {
			activateThisNodeOnly(n);
			});

		connect(autoSyncAct, &QAction::triggered, this, [=](bool checked) {
			node->setAutoSync(checked);
		});

		connect(enableDiscendants, &QAction::triggered, this, [&]() {
			autoSyncAllDescendants(n, true);
			});

		connect(disableDiscendants, &QAction::triggered, this, [&]() {
			autoSyncAllDescendants(n, false);
			});

		connect(enableAutoSync, &QAction::triggered, this, [=]() {
			autoSyncAllNodes(true);
			});

		connect(disableAutoSync, &QAction::triggered, this, [=]() {
			autoSyncAllNodes(false);
			});

		connect(delAct, &QAction::triggered, this, [&]() { this->removeNode(n); });
		connect(helpAct, &QAction::triggered, this, [&]() { this->showHelper(n); });

		menu->move(QCursor().pos().x() + 4, QCursor().pos().y() + 4);
		menu->show();
	}

	void QtNodeFlowScene::showThisNodeOnly(QtNode& n)
	{
		auto nodes = this->allNodes();
		for (auto node : nodes)
		{
			if (node->id() == n.id())
			{
				this->enableRendering(*node, false);
			}
			else
			{
				this->enableRendering(*node, true);
			}
		}

		this->updateNodeGraphView();

		nodes.clear();
	}

	void QtNodeFlowScene::showAllNodes()
	{
		auto nodes = this->allNodes();
		for (auto node : nodes)
		{
			this->enableRendering(*node, false);
		}

		this->updateNodeGraphView();

		nodes.clear();
	}

	void QtNodeFlowScene::activateThisNodeOnly(QtNode& n)
	{
		auto nodes = this->allNodes();
		for (auto node : nodes)
		{
			if (node->id() == n.id())
			{
				this->enablePhysics(*node, false);
			}
			else
			{
				this->enablePhysics(*node, true);
			}
		}

		this->updateNodeGraphView();

		nodes.clear();
	}

	void QtNodeFlowScene::activateAllNodes()
	{
		auto nodes = this->allNodes();
		for (auto node : nodes)
		{
			this->enablePhysics(*node, false);
		}

		this->updateNodeGraphView();

		nodes.clear();
	}

	void QtNodeFlowScene::autoSyncAllNodes(bool autoSync)
	{
		auto nodes = this->allNodes();
		for (auto node : nodes)
		{
			auto dataModel = dynamic_cast<QtNodeWidget*>(node->nodeDataModel());
			if (dataModel != nullptr)
			{
				auto dNode = dataModel->getNode();
				dNode->setAutoSync(autoSync);
			}
		}

		this->updateNodeGraphView();

		nodes.clear();
	}

	void QtNodeFlowScene::autoSyncAllDescendants(QtNode& n, bool autoSync)
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

		auto dataModel = dynamic_cast<QtNodeWidget*>(n.nodeDataModel());
		if (dataModel != nullptr)
		{
			auto dNode = dataModel->getNode();
			if (dNode == nullptr) return;

			scn->traverseForward(dNode, &act);
		}

		this->updateNodeGraphView();
	}

	//TODO: show a message on how to use this node
	void QtNodeFlowScene::showHelper(QtNode& n)
	{
		QMessageBox::information(nullptr, "Node Info", "Show something about this node");
	}

	void QtNodeFlowScene::reorderAllNodes()
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
				for (int i = 0; i < fieldInp.size(); i++)//����ÿ��Node��Inputfield
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
		std::map<dyno::ObjectId, QtNode*> qtNodeMapper;
		std::map<dyno::ObjectId, Node*> nodeMapper;
		for (auto const& _node : _nodes)
		{
			auto const& qtNode = _node.second;
			auto model = qtNode->nodeDataModel();

			auto nodeData = dynamic_cast<QtNodeWidget*>(model);

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
					QtNode* qtNode = qtNodeMapper[id];
					NodeGeometry& geo = qtNode->nodeGeometry();

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

		//��ɢ�ڵ������
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
				QtNode* qtNode = qtNodeMapper[id];
				NodeGeometry& geo = qtNode->nodeGeometry();
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
}
