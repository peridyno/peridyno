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
		if (node_widget != nullptr)
		{
			// Build virtual module
			dyno::Node* selectedNode = node_widget->getNode().get();
			QString str_vir = QString::fromStdString(selectedNode->getClassInfo()->getClassName() + "(virtual)");
			if (selectedNode != nullptr)
			{
				dyno::Object* obj = dyno::Object::createObject("VirtualModule<DataType3f>");
				dyno::Module* module_vir = dynamic_cast<dyno::Module*>(obj);
				if (module_vir != nullptr)
				{
					module_vir->setName(str_vir.toStdString());
					auto& fields = selectedNode->getAllFields();
					for (auto field : fields)
					{
						auto fType = field->getFieldType();
						if (fType == dyno::FieldTypeEnum::Current)
						{
							module_vir->addOutputField(field);
						}
					}
					QtDataModelRegistry::RegistryItemCreator creator = [str_vir, module_vir]() {
						auto dat = std::make_unique<QtModuleWidget>(module_vir);
						dat->setName(str_vir);
						return dat; };

					QString category = "Virtual";
					ret->registerModel<QtModuleWidget>(category, creator);
				}
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
		return;
		/*	clearScene();

			auto mlist = node->getModuleList();

			auto c = node->getAnimationPipeline()->entry();

			std::map<std::string, QtBlock*> moduleMap;

			int mSize = node->getAnimationPipeline()->size();


			auto addModuleWidget = [&](Module* m) -> void
			{
				auto module_name = m->getName();

				auto type = std::make_unique<QtNodes::QtModuleWidget>(m);

				auto& node = this->createNode(std::move(type));

				moduleMap[module_name] = &node;

				QPointF posView(m->bx(), m->by());

				node.nodeGraphicsObject().setPos(posView);

				this->nodePlaced(node);
			};

			addModuleWidget(node->getMechanicalState().get());

			for (; c != node->getAnimationPipeline()->finished(); c++)
			{
				addModuleWidget(c.get());
			}

			auto createModuleConnections = [&](Module* m) -> void
			{
				auto out_node = moduleMap[m->getName()];

				auto fields = m->getOutputFields();

				for (int i = 0; i < fields.size(); i++)
				{
					auto sink_fields = fields[i]->getSinks();
					for (int j = 0; j < sink_fields.size(); j++)
					{
						auto in_module = dynamic_cast<Module*>(sink_fields[j]->parent());
						if (in_module != nullptr)
						{
							auto in_fields = in_module->getInputFields();

							int in_port = -1;
							for (int t = 0; t < in_fields.size(); t++)
							{
								if (sink_fields[j] == in_fields[t])
								{
									in_port = t;
									break;
								}
							}

							if (in_port != -1)
							{
								auto in_node = moduleMap[in_module->getName()];

								createConnection(*in_node, in_port, *out_node, i);
							}
						}
					}
				}
			};*/

			//TODO: fix
		// 	createModuleConnections(node->getMechanicalState().get());
		// 	c = node->getAnimationPipeline()->entry();
		// 	for (; c != node->getAnimationPipeline()->finished(); c++)
		// 	{
		// 		createModuleConnections(c.get());
		// 	}
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
