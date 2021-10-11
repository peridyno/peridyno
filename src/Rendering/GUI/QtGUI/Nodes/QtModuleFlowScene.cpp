#include "QtModuleFlowScene.h"

#include "QtBlock.h"

#include "Object.h"
#include "SceneGraph.h"
#include "QtModuleWidget.h"

#include "SceneGraph.h"
#include "QtFlowView.h"
#include "DataModelRegistry.h"

#include "Module/VirtualModel.h"


namespace QtNodes
{

QtModuleFlowScene::QtModuleFlowScene(std::shared_ptr<DataModelRegistry> registry,
          QObject * parent)
  : QtFlowScene(registry, parent)
{
	connect(this, &QtFlowScene::nodeMoved, this, &QtModuleFlowScene::moveModulePosition);
}

QtModuleFlowScene::QtModuleFlowScene(QObject * parent, QtNodeWidget* node_widget)
	: QtFlowScene(parent)
{
	auto classMap = dyno::Object::getClassMap();

	auto ret = std::make_shared<QtNodes::DataModelRegistry>();
	int id = 0;
	for (auto const c : *classMap)
	{
		id++;

		QString str = QString::fromStdString(c.first);
		dyno::Object* obj = dyno::Object::createObject(str.toStdString());
		dyno::Module* module = dynamic_cast<dyno::Module*>(obj);

		if (module != nullptr)
		{
			QtNodes::DataModelRegistry::RegistryItemCreator creator = [str, module]() {
				auto dat = std::make_unique<QtNodes::QtModuleWidget>(module);
				dat->setName(str);
				return dat; };

			QString category = QString::fromStdString(module->getModuleType());
			ret->registerModel<QtNodes::QtModuleWidget>(category, creator);
		} 
	}
	if(node_widget != nullptr) 
	{
		// Build virtual module
		dyno::Node *selectedNode = node_widget->getNode().get();
		QString str_vir = QString::fromStdString(selectedNode->getClassInfo()->getClassName() + "(virtual)");
		if(selectedNode != nullptr)
		{
			dyno::Object* obj = dyno::Object::createObject("VirtualModel<DataType3f>");
			dyno::Module* module_vir = dynamic_cast<dyno::Module*>(obj);
			if(module_vir != nullptr)
			{
				module_vir->setName(str_vir.toStdString());
				auto& fields = selectedNode->getAllFields();
				for (auto field : fields)
				{
					auto fType = field->getFieldType();
					if(fType == dyno::FieldTypeEnum::Current)
					{
						module_vir->addOutputField(field);
					}
				}
				QtNodes::DataModelRegistry::RegistryItemCreator creator = [str_vir, module_vir]() {
					auto dat = std::make_unique<QtNodes::QtModuleWidget>(module_vir);
					dat->setName(str_vir);
					return dat; };
					
				QString category = "Virtual";
				ret->registerModel<QtNodes::QtModuleWidget>(category, creator);		
			}		
		}
	}
	

	this->setRegistry(ret);
}


QtModuleFlowScene::~QtModuleFlowScene()
{

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

void QtModuleFlowScene::moveModulePosition(QtBlock& n, const QPointF& newLocation)
{
	QtNodes::QtModuleWidget* mw = dynamic_cast<QtNodes::QtModuleWidget*>(n.nodeDataModel());

	Module* m = mw == nullptr ? nullptr : mw->getModule();

	if (m != nullptr)
	{
		m->setBlockCoord(newLocation.x(), newLocation.y());
	}
}

}
