#include "QtNodeFlowScene.h"

#include "QtBlock.h"
#include "QtFlowView.h"
#include "QtNodeWidget.h"

#include "Framework/Object.h"
#include "Framework/NodeIterator.h"
#include "Framework/NodePort.h"
#include "Framework/SceneGraph.h"

#include "DataModelRegistry.h"

#include <QtWidgets/QMessageBox>

namespace QtNodes
{

QtNodeFlowScene::QtNodeFlowScene(std::shared_ptr<DataModelRegistry> registry, QObject * parent)
  : QtFlowScene(registry, parent)
{

}

QtNodeFlowScene::QtNodeFlowScene(QObject * parent)
	: QtFlowScene(parent)
{
	auto classMap = dyno::Object::getClassMap();

	auto ret = std::make_shared<QtNodes::DataModelRegistry>();
	int id = 0;
	for (auto const c : *classMap)
	{
		id++;

		QString str = QString::fromStdString(c.first);
		auto obj = dyno::Object::createObject(str.toStdString());
		std::shared_ptr<Node> node(dynamic_cast<Node*>(obj));

		if (node != nullptr)
		{
			QtNodes::DataModelRegistry::RegistryItemCreator creator = [str]() {
				auto node_obj = dyno::Object::createObject(str.toStdString());
				std::shared_ptr<Node> new_node(dynamic_cast<Node*>(node_obj));
				auto dat = std::make_unique<QtNodeWidget>(std::move(new_node));
				return dat; 
			};

			QString category = "Default";// QString::fromStdString(module->getModuleType());
			ret->registerModel<QtNodeWidget>(category, creator);
		}
	}

	this->setRegistry(ret);

	dyno::SceneGraph& scn = dyno::SceneGraph::getInstance();
	showSceneGraph(&scn);

	connect(this, &QtFlowScene::nodeMoved, this, &QtNodeFlowScene::moveModulePosition);
}


QtNodeFlowScene::~QtNodeFlowScene()
{
	clearScene();
}


void QtNodeFlowScene::showSceneGraph(SceneGraph* scn)
{
	std::map<std::string, QtBlock*> nodeMap;

	auto root = scn->getRootNode();

	SceneGraph::Iterator it_end(nullptr);

	auto addNodeWidget = [&](std::shared_ptr<Node> m) -> void
	{
		auto module_name = m->getName();

		auto type = std::make_unique<QtNodeWidget>(m);

		auto& node = this->createNode(std::move(type));

		nodeMap[module_name] = &node;

		QPointF posView(m->bx(), m->by());

		node.nodeGraphicsObject().setPos(posView);

		this->nodePlaced(node);
	};

	for (auto it = scn->begin(); it != it_end; it++)
	{
		addNodeWidget(it.get());
	}

	auto createNodeConnections = [&](std::shared_ptr<Node> nd) -> void
	{
		auto in_name = nd->getName();
		
		if (nodeMap.find(in_name) != nodeMap.end())
		{
			auto in_block = nodeMap[nd->getName()];

			auto ports = nd->getAllNodePorts();

			for (int i = 0; i < ports.size(); i++)
			{
				dyno::NodePortType pType = ports[i]->getPortType();
				if (dyno::Single == pType)
				{
					auto node = ports[i]->getNodes()[0];
					if (node != nullptr)
					{
						auto in_block = nodeMap[node->getName()];
						createConnection(*in_block, 0, *in_block, i);
					}
				}
				else if (dyno::Multiple == pType)
				{
					//TODO: a weird problem exist here, if the expression "auto& nodes = ports[i]->getNodes()" is used,
					//we still have to call clear to avoid memory leak.
					auto nodes = ports[i]->getNodes();
					ports[i]->clear();
					for (int j = 0; j < nodes.size(); j++)
					{
						if (nodes[j] != nullptr)
						{
							auto out_name = nodes[j]->getName();
							if (nodeMap.find(out_name) != nodeMap.end())
							{
								auto out_block = nodeMap[nodes[j]->getName()];
								createConnection(*in_block, i, *out_block, 0);
							}
						}
					}
					nodes.clear();
				}
			}
		}
	};

	for (auto it = scn->begin(); it != it_end; it++)
	{
		createNodeConnections(it.get());
	}

// 	clearScene();
// 
	for (auto it = scn->begin(); it != it_end; it++)
	{
		auto node_ptr = it.get();
		std::cout << node_ptr->getClassInfo()->getClassName() << ": " << node_ptr.use_count() << std::endl;
	}
	nodeMap.clear();
}

void QtNodeFlowScene::moveModulePosition(QtBlock& n, const QPointF& newLocation)
{

}

}
