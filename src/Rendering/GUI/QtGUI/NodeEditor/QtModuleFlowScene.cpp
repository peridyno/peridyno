#include "QtModuleFlowScene.h"

#include "Object.h"
#include "SceneGraph.h"
#include "QtModuleWidget.h"
#include "SceneGraph.h"

#include "nodes/QNode"

#include "DirectedAcyclicGraph.h"
#include "AutoLayoutDAG.h"

namespace dyno
{
	class States : public Module
	{
		DECLARE_CLASS(States);
	public:
		States() {};
	};

	IMPLEMENT_CLASS(States);
}

namespace Qt
{
	QtModuleFlowScene::QtModuleFlowScene(std::shared_ptr<QtDataModelRegistry> registry,
		QObject* parent)
		: QtFlowScene(registry, parent)
	{
		connect(this, &QtFlowScene::nodeMoved, this, &QtModuleFlowScene::moveModule);
	}

	QtModuleFlowScene::QtModuleFlowScene(QObject* parent, QtNodeWidget* widget)
		: QtFlowScene(parent)
	{
		mNode = widget->getNode();

		auto classMap = dyno::Object::getClassMap();
		auto ret = std::make_shared<QtDataModelRegistry>();
		int id = 0;
		for (auto const c : *classMap)
		{
			id++;

			QString str = QString::fromStdString(c.first);
			dyno::Object* obj = dyno::Object::createObject(str.toStdString());
			std::shared_ptr<dyno::Module> module(dynamic_cast<dyno::Module*>(obj));
			//dyno::Module* module = dynamic_cast<dyno::Module*>(obj);

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

		if (mNode != nullptr)
			showModuleFlow(mNode.get());

		enableEditing();

		reorderAllModules();

		connect(this, &QtFlowScene::nodeMoved, this, &QtModuleFlowScene::moveModule);
		connect(this, &QtFlowScene::nodePlaced, this, &QtModuleFlowScene::addModule);
		connect(this, &QtFlowScene::nodeDeleted, this, &QtModuleFlowScene::deleteModule);
	}

	QtModuleFlowScene::~QtModuleFlowScene()
	{
		//To avoid editing the the module flow inside the node when the widget is closed.
		disableEditing();
	}

	void QtModuleFlowScene::enableEditing()
	{
		mEditingEnabled = true;

		auto& allNodes = this->allNodes();

		for each (auto node in allNodes)
		{
			auto model = dynamic_cast<QtModuleWidget*>(node->nodeDataModel());
			if (model != nullptr)
			{
				model->enableEditing();
			}
		}
	}

	void QtModuleFlowScene::disableEditing()
	{
		mEditingEnabled = false;

		auto& allNodes = this->allNodes();

		for each (auto node in allNodes)
		{
			auto model = dynamic_cast<QtModuleWidget*>(node->nodeDataModel());
			if (model != nullptr)
			{
				model->disableEditing();
			}
		}
	}

	void QtModuleFlowScene::showModuleFlow(Node* node)
	{
		clearScene();

		if (node == nullptr)
			return;

		auto mlist = node->getModuleList();

		std::map<dyno::ObjectId, QtNode*> moduleMap;

		//To show the animation pipeline in default
		if (mActivePipeline == nullptr)
			mActivePipeline = node->animationPipeline();

		auto& modules = mActivePipeline->allModules();


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

		//Create a dummy module to store all state variables
		mStates = std::make_shared<dyno::States>();
		auto& fields = node->getAllFields();
		for (auto field : fields)
		{
			if (field->getFieldType() == dyno::FieldTypeEnum::State)
			{
				mStates->addOutputField(field);
			}
		}

		addModuleWidget(mStates);

		for each (auto m in modules)
		{
			addModuleWidget(m.second);
		}

		auto createModuleConnections = [&](std::shared_ptr<Module> m) -> void
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
								nodeSrc = mStates.get();
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

		for each (auto m in modules)
		{
			createModuleConnections(m.second);
		}
	}

	void QtModuleFlowScene::moveModule(QtNode& n, const QPointF& newLocation)
	{
		QtModuleWidget* mw = dynamic_cast<QtModuleWidget*>(n.nodeDataModel());

		auto m = mw == nullptr ? nullptr : mw->getModule();

		if (m != nullptr)
		{
			m->setBlockCoord(newLocation.x(), newLocation.y());
		}
	}

	void QtModuleFlowScene::showAnimationPipeline()
	{
		if (mNode == nullptr)
			return;

		mActivePipeline = mNode->animationPipeline();

		updateModuleGraphView();
	}

	void QtModuleFlowScene::showGraphicsPipeline()
	{
		if (mNode == nullptr)
			return;

		mActivePipeline = mNode->graphicsPipeline();

		updateModuleGraphView();
	}

	void QtModuleFlowScene::updateModuleGraphView()
	{
		disableEditing();

		clearScene();

		if (mNode != nullptr)
			showModuleFlow(mNode.get());

		enableEditing();
	}

	void QtModuleFlowScene::reorderAllModules()
	{
		dyno::DirectedAcyclicGraph graph;

		auto constructDAG = [&](std::shared_ptr<Module> m) -> void
		{
			auto outId = m->objectId();

			auto fieldOut = m->getOutputFields();
			for (int i = 0; i < fieldOut.size(); i++)
			{
				auto& sinks = fieldOut[i]->getSinks();
				for each (auto sink in sinks)
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

		constructDAG(mStates);

		auto& mlists = mNode->animationPipeline()->activeModules();
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

		//DOTO: optimize the position for the dummy module
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

	void QtModuleFlowScene::addModule(QtNode& n)
	{
		auto nodeData = dynamic_cast<QtModuleWidget*>(n.nodeDataModel());

		if (mEditingEnabled && nodeData != nullptr) {
			mNode->animationPipeline()->pushModule(nodeData->getModule());
		}
	}

	void QtModuleFlowScene::deleteModule(QtNode& n)
	{
		auto nodeData = dynamic_cast<QtModuleWidget*>(n.nodeDataModel());

		if (mEditingEnabled && nodeData != nullptr) {
			mNode->animationPipeline()->popModule(nodeData->getModule());
		}
	}
}
