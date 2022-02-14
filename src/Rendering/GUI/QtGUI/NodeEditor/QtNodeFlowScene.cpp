#include "QtNodeFlowScene.h"
#include "QtNodeWidget.h"

#include "nodes/QNode"

#include "Object.h"
#include "NodeIterator.h"
#include "NodePort.h"
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

				QString category = "Default";// QString::fromStdString(module->getModuleType());
				ret->registerModel<QtNodeWidget>(category, creator);
			}
		}

		this->setRegistry(ret);

		auto scn = dyno::SceneGraphFactory::instance()->active();
		showSceneGraph(scn.get());

		connect(this, &QtFlowScene::nodeMoved, this, &QtNodeFlowScene::moveModulePosition);
		connect(this, &QtFlowScene::nodePlaced, this, &QtNodeFlowScene::addNodeToSceneGraph);
		connect(this, &QtFlowScene::nodeDeleted, this, &QtNodeFlowScene::deleteNodeToSceneGraph);
	}


	QtNodeFlowScene::~QtNodeFlowScene()
	{
		clearScene();
	}


	void QtNodeFlowScene::showSceneGraph(SceneGraph* scn)
	{
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

		// 	clearScene();
		// 
		for (auto it = scn->begin(); it != scn->end(); it++)
		{
			auto node_ptr = it.get();
			std::cout << node_ptr->getClassInfo()->getClassName() << ": " << node_ptr.use_count() << std::endl;
		}
		nodeMap.clear();
	}

	void QtNodeFlowScene::moveModulePosition(QtNode& n, const QPointF& newLocation)
	{

	}

	void QtNodeFlowScene::addNodeToSceneGraph(QtNode& n)
	{
		auto nodeData = dynamic_cast<QtNodeWidget*>(n.nodeDataModel());

		printf("Use count before add: %d \n", nodeData->getNode().use_count());

		if (nodeData != nullptr) {
			auto scn = dyno::SceneGraphFactory::instance()->active();
			scn->addNode(nodeData->getNode());
		}

		printf("Use count after add: %d \n", nodeData->getNode().use_count());
	}

	void QtNodeFlowScene::deleteNodeToSceneGraph(QtNode& n)
	{
		auto nodeData = dynamic_cast<QtNodeWidget*>(n.nodeDataModel());

		printf("Use count before: %d \n", nodeData->getNode().use_count());
		if (nodeData != nullptr) {
			auto scn = dyno::SceneGraphFactory::instance()->active();
			scn->deleteNode(nodeData->getNode());
		}

		printf("Use count after: %d \n", nodeData->getNode().use_count());
	}

}
