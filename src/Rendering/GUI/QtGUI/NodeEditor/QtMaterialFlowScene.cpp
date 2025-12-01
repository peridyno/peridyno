#include "QtMaterialFlowScene.h"

#include "Object.h"
#include "SceneGraph.h"
#include "QtModuleWidget.h"
#include "SceneGraph.h"

#include "nodes/QNode"

#include "DirectedAcyclicGraph.h"
#include "AutoLayoutDAG.h"
#include "ModulePort.h"

#include <QtWidgets/QMenu>
#include <QLineEdit>
#include <QtWidgets>

namespace dyno
{
	class States : public Module
	{
	public:
		States() {};

		bool allowExported() override { return false; }
		bool allowImported() override { return false; }

		std::string caption() { return "States"; }
	};

	class Outputs : public Module
	{
	public:
		Outputs() {};

		bool allowExported() override { return false; }
		bool allowImported() override { return false; }

		std::string caption() { return "Outputs"; }
	};
}

namespace Qt
{
	QPointF MatStatePos = QPointF(0.0f, 0.0f);

	QtMaterialFlowScene::QtMaterialFlowScene(std::shared_ptr<QtDataModelRegistry> registry,
		QObject* parent)
		: QtFlowScene(registry, parent)
	{
		connect(this, &QtFlowScene::nodeMoved, this, &QtMaterialFlowScene::moveModule);
	}

	QtMaterialFlowScene::QtMaterialFlowScene(std::shared_ptr<CustomMaterial> customMaterial, QObject* parent)
		: QtFlowScene(parent)
	{
		mCustomMaterial = customMaterial;
		if (mCustomMaterial)
			mMaterialPipline = mCustomMaterial->materialPipeline();
		auto classMap = dyno::Object::getClassMap();
		auto ret = std::make_shared<QtDataModelRegistry>();
		int id = 0;
		for (auto const c : *classMap)
		{
			id++;

			QString str = QString::fromStdString(c.first);
			dyno::Object* obj = dyno::Object::createObject(str.toStdString());
			std::shared_ptr<dyno::Module> module(dynamic_cast<dyno::Module*>(obj));

			if (module != nullptr)
			{
				QtDataModelRegistry::RegistryItemCreator creator = [str]() {
					auto new_obj = dyno::Object::createObject(str.toStdString());
					std::shared_ptr<dyno::Module> new_module(dynamic_cast<dyno::Module*>(new_obj));
					auto dat = std::make_unique<QtModuleWidget>(new_module);
					return dat; 
				};

				QString category = QString::fromStdString(module->getModuleType());
				ret->registerModel<QtModuleWidget>(category, creator);
			}
		}

		this->setRegistry(ret);

		if (mCustomMaterial != nullptr)
			showMaterialFlow(mCustomMaterial);

		enableEditing();

		reorderAllModules();

		connect(this, &QtFlowScene::nodeMoved, this, &QtMaterialFlowScene::moveModule);
		connect(this, &QtFlowScene::nodePlaced, this, &QtMaterialFlowScene::addModule);
		connect(this, &QtFlowScene::nodeDeleted, this, &QtMaterialFlowScene::deleteModule);

		//connect(this, &QtFlowScene::outPortConextMenu, this, &QtMaterialFlowScene::promoteOutput);
	}

	QtMaterialFlowScene::~QtMaterialFlowScene()
	{
		//To avoid editing the the module flow inside the node when the widget is closed.
		disableEditing();
	}

	void QtMaterialFlowScene::enableEditing()
	{
		mEditingEnabled = true;

		auto allNodes = this->allNodes();

		for  (auto node : allNodes)
		{
			auto model = dynamic_cast<QtModuleWidget*>(node->nodeDataModel());
			if (model != nullptr)
			{
				model->enableEditing();
			}
		}
	}

	void QtMaterialFlowScene::disableEditing()
	{
		mEditingEnabled = false;

		auto allNodes = this->allNodes();

		for  (auto node : allNodes)
		{
			auto model = dynamic_cast<QtModuleWidget*>(node->nodeDataModel());
			if (model != nullptr)
			{
				model->disableEditing();
			}
		}
	}

	void QtMaterialFlowScene::showMaterialFlow(std::shared_ptr<CustomMaterial> customMaterial)
	{
		clearScene();

		if (customMaterial == nullptr)
			return;

		auto allModules = customMaterial->piplineModules();

		std::map<dyno::ObjectId, QtNode*> moduleMap;

		auto& modules = customMaterial->materialPipeline()->allModules();

		auto addModuleWidget = [&](std::shared_ptr<Module> m) -> void
		{
			auto mId = m->objectId();

			auto type = std::make_unique<QtModuleWidget>(m);

			auto& node = this->createNode(std::move(type));

			moduleMap[mId] = &node;

			QPointF posView(m->bx(), m->by());

			node.nodeGraphicsObject().setPos(posView);

			this->nodePlaced(node);
		};

		for  (auto m : modules)
		{
			addModuleWidget(m.second);
		}

		auto createModuleConnections = [&](std::shared_ptr<Module> m) -> void
		{
			auto inId = m->objectId();

			if (moduleMap.find(inId) != moduleMap.end()) {
				auto inBlock = moduleMap[m->objectId()];

				auto imports = m->getImportModules();

				if (m->allowImported())
				{
					for (int i = 0; i < imports.size(); i++)
					{
						dyno::ModulePortType pType = imports[i]->getPortType();
						if (dyno::Single == pType)
						{
							auto moduleSrc = imports[i]->getModules()[0];
							if (moduleSrc != nullptr)
							{
								auto outId = moduleSrc->objectId();
								if (moduleMap.find(outId) != moduleMap.end())
								{
									auto outBlock = moduleMap[moduleSrc->objectId()];
									createConnection(*inBlock, i, *outBlock, 0);
								}
							}
						}
						else if (dyno::Multiple == pType)
						{
							//TODO: a weird problem exist here, if the expression "auto& nodes = ports[i]->getNodes()" is used,
							//we still have to call clear to avoid memory leak.
							auto& modules = imports[i]->getModules();
							//ports[i]->clear();
							for (int j = 0; j < modules.size(); j++)
							{
								if (modules[j] != nullptr)
								{
									auto outId = modules[j]->objectId();
									if (moduleMap.find(outId) != moduleMap.end())
									{
										auto outBlock = moduleMap[outId];
										createConnection(*inBlock, i, *outBlock, 0);
									}
								}
							}
							//nodes.clear();
						}
					}
				}

				auto fieldIn = m->getInputFields();

				for (int i = 0; i < fieldIn.size(); i++)
				{
					auto fieldSrc = fieldIn[i]->getSource();
					if (fieldSrc != nullptr) {
						auto parSrc = fieldSrc->parent();
						if (parSrc != nullptr)
						{
							Module* moduleSrc = dynamic_cast<Module*>(parSrc);
							if (moduleSrc == nullptr)
							{
								continue;
							}

							auto outId = moduleSrc->objectId();
							auto fieldsOut = moduleSrc->getOutputFields();

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

							if (moduleSrc->allowExported()) outFieldIndex++;

							uint inFieldIndex = m->allowImported() ? i + imports.size() : i;

							if (fieldFound && moduleMap.find(outId) != moduleMap.end())
							{
								auto outBlock = moduleMap[outId];
								createConnection(*inBlock, inFieldIndex, *outBlock, outFieldIndex);
							}
						}
					}
				}

			}
		};

		for  (auto m : modules)
		{
			createModuleConnections(m.second);
		}

	}

	void QtMaterialFlowScene::moveModule(QtNode& n, const QPointF& newLocation)
	{
		QtModuleWidget* mw = dynamic_cast<QtModuleWidget*>(n.nodeDataModel());

		auto m = mw == nullptr ? nullptr : mw->getModule();

		if (m != nullptr)
		{
			m->setBlockCoord(newLocation.x(), newLocation.y());
		}
	}


	void QtMaterialFlowScene::showCustomMaterialPipeline()
	{
		auto pipeline = mCustomMaterial->materialPipeline();

		if (mCustomMaterial == nullptr)
			return;

		updateModuleGraphView();

		if (mReorderGraphicsPipeline) {
			reorderAllModules();
			mReorderGraphicsPipeline = false;
		}
	}

	void QtMaterialFlowScene::reconstructActivePipeline()
	{
		mMaterialPipline->forceUpdate();
	}

	void QtMaterialFlowScene::promoteOutput(QtNode& n, const PortIndex index, const QPointF& pos)
	{
		//Index 0 is used for module connection
		if (index == 0)
			return;

		QMenu portMenu;

		portMenu.setObjectName("PortMenu");

		QPoint p(pos.x(), pos.y());

		portMenu.exec(p);
	}

	void QtMaterialFlowScene::updateModuleGraphView()
	{
		disableEditing();

		clearScene();

		if (mCustomMaterial != nullptr)
			showMaterialFlow(mCustomMaterial);

		enableEditing();
	}

	void QtMaterialFlowScene::reorderAllModules()
	{
		if (mCustomMaterial == nullptr)
			return;

		dyno::DirectedAcyclicGraph graph;

		auto constructDAG = [&](std::shared_ptr<Module> m) -> void
		{
			if (m->allowImported())
			{
				auto imports = m->getImportModules();
				for (int i = 0; i < imports.size(); i++)
				{
					auto inId = m->objectId();
					dyno::ModulePortType pType = imports[i]->getPortType();
					if (dyno::Single == pType)
					{
						auto module = imports[i]->getModules()[0];
						if (module != nullptr)
						{
							auto outId = module->objectId();

							graph.addEdge(outId, inId);
						}
					}
					else if (dyno::Multiple == pType)
					{
						auto& modules = imports[i]->getModules();
						for (int j = 0; j < modules.size(); j++)
						{
							if (modules[j] != nullptr)
							{
								auto outId = modules[j]->objectId();

								graph.addEdge(outId, inId);
							}
						}
						//nodes.clear();
					}
				}
			}

			auto outId = m->objectId();
			auto fieldOut = m->getOutputFields();
			for (int i = 0; i < fieldOut.size(); i++)
			{
				auto& sinks = fieldOut[i]->getSinks();
				for  (auto sink : sinks)
				{
					if (sink != nullptr) {
						auto parSrc = sink->parent();
						if (parSrc != nullptr)
						{
							Module* nodeSrc = dynamic_cast<Module*>(parSrc);

							if (nodeSrc != nullptr)
							{
								auto inId = nodeSrc->objectId();
								graph.addEdge(outId, inId);
							}
						}
					}
				}
			}
		};

		auto& mlists = mMaterialPipline->activeModules();
		for (auto it = mlists.begin(); it != mlists.end(); it++)
		{
			constructDAG(*it);
		}

		dyno::AutoLayoutDAG layout(&graph);
		layout.update();

		//Set up the mapping from ObjectId to QtNode
		auto& _nodes = this->nodes();
		std::map<dyno::ObjectId, QtNode*> qtNodeMapper;
		std::map<dyno::ObjectId, Module*> moduleMapper;
		for (auto const& _node : _nodes)
		{
			auto const& qtNode = _node.second;
			auto model = qtNode->nodeDataModel();

			auto nodeData = dynamic_cast<QtModuleWidget*>(model);

			if (model != nullptr)
			{
				auto m = nodeData->getModule();
				if (m != nullptr)
				{
					qtNodeMapper[m->objectId()] = qtNode.get();
					moduleMapper[m->objectId()] = m.get();
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

					Module* node = moduleMapper[id];

					node->setBlockCoord(offsetX, offsetY);

					offsetY += (h + mDy);
				}
			}

			offsetX += (xMax + mDx);
		}
			

		qtNodeMapper.clear();
		moduleMapper.clear();

		updateModuleGraphView();
	}

	void QtMaterialFlowScene::addModule(QtNode& n)
	{
		if (mMaterialPipline == nullptr)
			return;

		auto nodeData = dynamic_cast<QtModuleWidget*>(n.nodeDataModel());

		if (mEditingEnabled && nodeData != nullptr) {
			mMaterialPipline->pushModule(nodeData->getModule());
		}
	}

	void QtMaterialFlowScene::deleteModule(QtNode& n)
	{
		if (mMaterialPipline == nullptr)
			return;

		auto nodeData = dynamic_cast<QtModuleWidget*>(n.nodeDataModel());

		if (mEditingEnabled && nodeData != nullptr) {
			mMaterialPipline->popModule(nodeData->getModule());

			emit this->nodeDeselected();
		}
	}
}
