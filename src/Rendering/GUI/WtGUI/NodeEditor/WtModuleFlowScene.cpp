#include "AutoLayoutDAG.h"
#include "WtModuleFlowScene.h"
#include "WtNodeWidget.h"
#include <DirectedAcyclicGraph.h>

#include "Object.h"

namespace dyno
{
	class WtStates : public Module
	{
		DECLARE_CLASS(WtStates);
	public:
		WtStates() {};
	};

	IMPLEMENT_CLASS(WtStates);
}

Wt::WPointF SimStatePos = Wt::WPointF(0.0f, 0.0f);
Wt::WPointF RenStatePos = Wt::WPointF(0.0f, 0.0f);

WtModuleFlowScene::WtModuleFlowScene(Wt::WPainter* painter, std::shared_ptr<dyno::Node> node , PipelineType pipelineType)
	: _painter(painter)
	, mNode(node)
	, mPipelineType(pipelineType)
{
	if (node != nullptr)
		showModuleFlow(mNode);

	//reorderAllModules();
}

WtModuleFlowScene::~WtModuleFlowScene()
{
}

void WtModuleFlowScene::enableEditing()
{
	mEditingEnabled = true;

	auto allNodes = this->allNodes();

	for (auto node : allNodes)
	{
		auto model = dynamic_cast<WtModuleWidget*>(node->nodeDataModel());
		if (model != nullptr)
		{
			model->enableEditing();
		}
	}
}

void WtModuleFlowScene::disableEditing()
{
	mEditingEnabled = false;

	auto allNodes = this->allNodes();

	for (auto node : allNodes)
	{
		auto model = dynamic_cast<WtModuleWidget*>(node->nodeDataModel());
		if (model != nullptr)
		{
			model->disableEditing();
		}
	}
}

void WtModuleFlowScene::updateModuleGraphView()
{
	//disableEditing();

	clearScene();

	if (mNode != nullptr)
		showModuleFlow(mNode);

	//enableEditing();
}

void WtModuleFlowScene::reorderAllModules()
{
	if (mActivePipeline == nullptr)
		return;

	dyno::DirectedAcyclicGraph graph;

	auto constructDAG = [&](std::shared_ptr<dyno::Module> m) -> void
		{
			auto outId = m->objectId();

			auto fieldOut = m->getOutputFields();

			for (int i = 0; i < fieldOut.size(); i++)
			{
				auto& sinks = fieldOut[i]->getSinks();
				for (auto sink : sinks)
				{
					if (sink != nullptr)
					{
						auto parSrc = sink->parent();
						if (parSrc != nullptr)
						{
							dyno::Module* nodeSrc = dynamic_cast<dyno::Module*>(parSrc);

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

	auto& mlists = mActivePipeline->activeModules();
	for (auto it = mlists.begin(); it != mlists.end(); it++)
		constructDAG(*it);

	dyno::AutoLayoutDAG layout(&graph);
	layout.update();

	//Set up the mapping from ObjectId to WtNode
	auto& _nodes = this->nodes();
	std::map<dyno::ObjectId, WtNode*> wtNodeMapper;
	std::map<dyno::ObjectId, dyno::Module*> moduleMapper;
	for (auto const& _node : _nodes)
	{
		auto const& wtNode = _node.second;
		auto model = wtNode->nodeDataModel();

		auto nodeData = dynamic_cast<WtModuleWidget*>(model);

		if (model != nullptr)
		{
			auto m = nodeData->getModule();
			if (m != nullptr)
			{
				wtNodeMapper[m->objectId()] = wtNode.get();
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
			if (wtNodeMapper.find(id) != wtNodeMapper.end())
			{
				WtNode* wtNode = wtNodeMapper[id];
				WtNodeGeometry& geo = wtNode->nodeGeometry();

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

	if (mActivePipeline == mNode->animationPipeline()) {
		SimStatePos = Wt::WPointF(mStates->bx(), mStates->by());
	}
	else {
		RenStatePos = Wt::WPointF(mStates->bx(), mStates->by());
	}

	wtNodeMapper.clear();
	moduleMapper.clear();

	updateModuleGraphView();
}

void WtModuleFlowScene::showModuleFlow(std::shared_ptr<dyno::Node> node)
{
	//clearScene();

	if (node == nullptr)
		return;

	//auto& mlist = node->getModuleList();

	std::map<dyno::ObjectId, WtNode*> moduleMap;

	// To show the animation pipeline
	if (mActivePipeline == nullptr)
	{
		if (mPipelineType ==PipelineType::Reset)
		{
			mActivePipeline = node->resetPipeline();
		}
		else if (mPipelineType == PipelineType::Animation)
		{
			mActivePipeline = node->animationPipeline();
		}
		else if (mPipelineType == PipelineType::Graphics)
		{
			mActivePipeline = node->graphicsPipeline();
		}
	}

	auto& modules = mActivePipeline->allModules();

	auto addModuleWidget = [&](std::shared_ptr<dyno::Module> m) -> void
		{
			auto mId = m->objectId();

			auto type = std::make_unique<WtModuleWidget>(m);

			auto& node = this->createNode(std::move(type), _painter, -1);

			node.setModule(m);

			moduleMap[mId] = OutNodeMap[mId] = &node;

			Wt::WPointF posView(m->bx(), m->by());

			node.nodeGraphicsObject().setPos(posView);
			//this->nodePlaced(node);
		};

	//Create a dummy module to store all state variables
	mStates = std::make_shared<dyno::WtStates>();

	auto& fields = node->getAllFields();
	for (auto field : fields)
	{
		if (field->getFieldType() == dyno::FieldTypeEnum::State
			|| field->getFieldType() == dyno::FieldTypeEnum::In)
		{
			mStates->addOutputField(field);
		}
	}

	Wt::WPointF pos = mActivePipeline == node->animationPipeline() ? SimStatePos : RenStatePos;
	mStates->setBlockCoord(pos.x(), pos.y());

	addModuleWidget(mStates);

	for (auto m : modules)
	{
		addModuleWidget(m.second);
	}

	auto createModuleConnections = [&](std::shared_ptr<dyno::Module> m)->void
		{
			auto inId = m->objectId();

			if (moduleMap.find(inId) != moduleMap.end())
			{
				auto inBlock = moduleMap[m->objectId()];

				auto fieldIn = m->getInputFields();

				for (int i = 0; i < fieldIn.size(); i++)
				{
					auto fieldSrc = fieldIn[i]->getSource();
					if (fieldSrc != nullptr)
					{
						auto parSrc = fieldSrc->parent();
						if (parSrc != nullptr)
						{
							dyno::Module* nodeSrc = dynamic_cast<dyno::Module*>(parSrc);
							if (nodeSrc == nullptr)
								nodeSrc = mStates.get();

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

							if (fieldFound && moduleMap.find(outId) != moduleMap.end())
							{
								auto outBlock = moduleMap[outId];
								createConnection(*inBlock, i, *outBlock, outFieldIndex, _painter);
								addConnection(inBlock->getModule(), outBlock->getModule());
							}
						}
					}
				}
			}
		};

	for (auto m : modules)
	{
		createModuleConnections(m.second);
	}
}

void WtModuleFlowScene::addModule(std::shared_ptr<dyno::Module> new_module)
{
	if (mActivePipeline == nullptr)
		return;

	if (mEditingEnabled && new_module != nullptr)
	{
		mActivePipeline->pushModule(new_module);
	}
}

void WtModuleFlowScene::deleteModule(std::shared_ptr<dyno::Module> delete_module)
{
	if (mActivePipeline == nullptr)
		return;

	if (mEditingEnabled && delete_module != nullptr)
	{
		mActivePipeline->popModule(delete_module);
	}
}

void WtModuleFlowScene::showResetPipeline()
{
	auto pipeline = mNode->resetPipeline();

	if (mNode == nullptr || mActivePipeline == pipeline)
		return;

	mActivePipeline = pipeline;

	updateModuleGraphView();
}

void WtModuleFlowScene::showAnimationPipeline()
{
	auto pipeline = mNode->animationPipeline();

	if (mNode == nullptr || mActivePipeline == pipeline)
		return;

	mActivePipeline = pipeline;

	updateModuleGraphView();
}

void WtModuleFlowScene::showGraphicsPipeline()
{
	auto pipeline = mNode->graphicsPipeline();

	if (mNode == nullptr || mActivePipeline == pipeline)
		return;

	mActivePipeline = pipeline;

	updateModuleGraphView();
}

void WtModuleFlowScene::addConnection(std::shared_ptr<dyno::Module> exportModule, std::shared_ptr<dyno::Module> inportModule)
{
	connectionData temp;
	temp.exportModule = exportModule;
	temp.inportModule = inportModule;
	nodeConnections.push_back(temp);
}
