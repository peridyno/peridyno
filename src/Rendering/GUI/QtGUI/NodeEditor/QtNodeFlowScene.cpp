#include "QtNodeFlowScene.h"
#include "QtNodeWidget.h"

#include "nodes/QNode"

#include "Object.h"
#include "NodeIterator.h"
#include "NodePort.h"
#include "Action.h"
#include "DirectedAcyclicGraph.h"
#include "AutoLayoutDAG.h"
#include "SceneGraphFactory.h"

#include <QtWidgets/QMessageBox>

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

				QString category = QString::fromStdString(node->getNodeType());
				ret->registerModel<QtNodeWidget>(category, creator);
			}
		}
	
		this->setRegistry(ret);

		createNodeGraphView();
		reorderAllNodes();

		connect(this, &QtFlowScene::nodeMoved, this, &QtNodeFlowScene::moveNode);
		connect(this, &QtFlowScene::nodePlaced, this, &QtNodeFlowScene::addNode);
		connect(this, &QtFlowScene::nodeDeleted, this, &QtNodeFlowScene::deleteNode);

		connect(this, &QtFlowScene::nodeHotKey0Checked, this, &QtNodeFlowScene::enableRendering);
		connect(this, &QtFlowScene::nodeHotKey1Checked, this, &QtNodeFlowScene::enablePhysics);
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

			this->nodePlaced(node);
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
							Node* nodeSrc = dynamic_cast<Node*>(parSrc);

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

							if (fieldFound && nodeMap.find(outId) != nodeMap.end())
							{
								auto outBlock = nodeMap[outId];
								createConnection(*inBlock, i + ports.size(), *outBlock, 1 + outFieldIndex);
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
			if(x== mId){
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

		auto& allNodes = this->allNodes();

		for each (auto node in allNodes)
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

		auto& allNodes = this->allNodes();

		for each (auto node in allNodes)
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
		}
	}

	void QtNodeFlowScene::dynoNodePlaced(std::shared_ptr<dyno::Node> node)
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
			node->setVisible(checked);
		}
	}

	void QtNodeFlowScene::enablePhysics(QtNode& n, bool checked)
	{
		auto nodeData = dynamic_cast<QtNodeWidget*>(n.nodeDataModel());

		if (mEditingEnabled && nodeData != nullptr) {
			auto node = nodeData->getNode();
			//node->setActive(checked);
			if (checked)
				node->animationPipeline()->enable();
			else
				node->animationPipeline()->disable();
		}
	}

	void QtNodeFlowScene::reorderAllNodes()
	{
		auto scn = dyno::SceneGraphFactory::instance()->active();

		dyno::DirectedAcyclicGraph graph;

		auto constructDAG = [&](std::shared_ptr<Node> nd) -> void
		{
			auto inId = nd->objectId();

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
						
						graph.addEdge(outId, inId);
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
						Node* nodeSrc = dynamic_cast<Node*>(parSrc);

						auto outId = nodeSrc->objectId();
						
						graph.addEdge(outId, inId);
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
		}

		qtNodeMapper.clear();
		nodeMapper.clear();

		updateNodeGraphView();
	}
}
