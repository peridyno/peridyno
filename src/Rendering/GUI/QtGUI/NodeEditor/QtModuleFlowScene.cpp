#include "QtModuleFlowScene.h"

#include "Object.h"
#include "SceneGraph.h"
#include "QtModuleWidget.h"
#include "SceneGraph.h"

#include "Module/VirtualModule.h"

#include "nodes/QNode"

namespace Qt
{
	QtModuleFlowScene::QtModuleFlowScene(std::shared_ptr<QtDataModelRegistry> registry,
		QObject* parent)
		: QtFlowScene(registry, parent)
	{
		connect(this, &QtFlowScene::nodeMoved, this, &QtModuleFlowScene::moveModulePosition);
	}

	QtModuleFlowScene::QtModuleFlowScene(QObject* parent, QtNodeWidget* node_widget)
		: QtFlowScene(parent)
	{
		auto classMap = dyno::Object::getClassMap();
		m_parent_node = node_widget;
		auto ret = std::make_shared<QtDataModelRegistry>();
		int id = 0;
		for (auto const c : *classMap)
		{
			id++;

			QString str = QString::fromStdString(c.first);
			dyno::Object* obj = dyno::Object::createObject(str.toStdString());
			dyno::Module* module = dynamic_cast<dyno::Module*>(obj);

			if (module != nullptr)
			{
				QtDataModelRegistry::RegistryItemCreator creator = [str, module]() {
					auto dat = std::make_unique<QtModuleWidget>(module);
					dat->setName(str);
					return dat; };

				QString category = QString::fromStdString(module->getModuleType());
				ret->registerModel<QtModuleWidget>(category, creator);
			}
		}

		this->setRegistry(ret);
	}


	QtModuleFlowScene::~QtModuleFlowScene()
	{

	}


	void QtModuleFlowScene::pushModule()
	{
		if (m_parent_node == nullptr)
		{
			return;
		}
		dyno::Node* selectedNode = m_parent_node->getNode().get();
		// clear
		// selectedNode->graphicsPipeline()->clear();

		// push
		auto const& nodes = this->nodes();
		for (auto const& pair : nodes)
		{
			auto const& node = pair.second;
			auto const& module_widget = dynamic_cast<QtModuleWidget*>(node->nodeDataModel());
			if (module_widget == nullptr)
			{
				continue;
			}

			auto const& module = module_widget->getModule();
			if (module == nullptr)
			{
				continue;
			}

			std::string class_name = module->getClassInfo()->getClassName();
			if (class_name.find("virtual") == std::string::npos)
			{
				selectedNode->graphicsPipeline()->pushModule(std::shared_ptr<dyno::Module>(module));
			}
		}
	}

	void QtModuleFlowScene::showNodeFlow(Node* node)
	{
			clearScene();

			auto mlist = node->getModuleList();

			std::map<dyno::ObjectId, QtNode*> moduleMap;

			auto& activeModules = node->animationPipeline()->activeModules();


			auto addModuleWidget = [&](Module* m) -> void
			{
				auto mId = m->objectId();

				auto type = std::make_unique<QtModuleWidget>(m);

				auto& node = this->createNode(std::move(type));

				moduleMap[mId] = &node;

				QPointF posView(m->bx(), m->by());

				node.nodeGraphicsObject().setPos(posView);

				this->nodePlaced(node);
			};

			//Add a virtual module
			//addModuleWidget(node->getMechanicalState().get());

			//Create a virtual module
			Module* states = new Module;
			auto& fields = node->getAllFields();
			for (auto field : fields)
			{
				if (field->getFieldType() == dyno::FieldTypeEnum::State)
				{
					states->addOutputField(field);
				}
			}

			addModuleWidget(states);

			for each (auto m in activeModules)
			{
				addModuleWidget(m);
			}

			auto createModuleConnections = [&](Module* m) -> void
			{
				auto inId = m->objectId();

				if (moduleMap.find(inId) != moduleMap.end()) {
					auto inBlock = moduleMap[m->objectId()];

					auto fieldIn = m->getInputFields();

					for (int i = 0; i < fieldIn.size(); i++)
					{
						auto fieldSrc = fieldIn[i]->getSource();
						if (fieldSrc != nullptr) {
							auto parSrc = fieldSrc->parent();
							if (parSrc != nullptr)
							{
								Module* nodeSrc = dynamic_cast<Module*>(parSrc);
								if (nodeSrc == nullptr)
								{
									nodeSrc = states;
								}

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

								if (fieldFound && moduleMap.find(outId) != moduleMap.end())
								{
									auto outBlock = moduleMap[outId];
									createConnection(*inBlock, i, *outBlock, outFieldIndex);
								}
							}
						}
					}
				}
			};

			auto rit = activeModules.rbegin();
			while (rit != activeModules.rend()) {
				createModuleConnections(*rit);
				rit++;
			}
	}

	void QtModuleFlowScene::moveModulePosition(QtNode& n, const QPointF& newLocation)
	{
		QtModuleWidget* mw = dynamic_cast<QtModuleWidget*>(n.nodeDataModel());

		Module* m = mw == nullptr ? nullptr : mw->getModule();

		if (m != nullptr)
		{
			m->setBlockCoord(newLocation.x(), newLocation.y());
		}
	}
}
