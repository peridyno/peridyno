#include "WtModuleFlowScene.h"
#include "WtNodeWidget.h"

Wt::WPointF SimStatePos = Wt::WPointF(0.0f, 0.0f);
Wt::WPointF RenStatePos = Wt::WPointF(0.0f, 0.0f);

WtModuleFlowScene::WtModuleFlowScene(Wt::WPainter* painter, std::shared_ptr<dyno::Node> node, std::shared_ptr<dyno::SceneGraph> scene)
	: _painter(painter)
	, mNode(node)
	, mScene(scene)
{
	showModuleFlow(node);
}

WtModuleFlowScene::~WtModuleFlowScene()
{
}

void WtModuleFlowScene::showModuleFlow(std::shared_ptr<dyno::Node> node)
{
	//clearScene();

	if (node == nullptr)
		return;

	auto& mlist = node->getModuleList();

	std::map<dyno::ObjectId, WtNode*> moduleMpa;

	// To show the animation pipeline
	if (mActivePipeline == nullptr)
		mActivePipeline = node->graphicsPipeline();

	auto& modules = mActivePipeline->allModules();

	auto addModuleWidget = [&](std::shared_ptr<dyno::Module> m) -> void
		{
			auto mId = m->objectId();

			auto type = std::make_unique<WtModuleWidget>(m);

			auto& node = this->createNode(std::move(type), _painter, -1);

			moduleMpa[mId] = &node;

			Wt::WPointF posView(m->bx(), m->by());

			node.nodeGraphicsObject().setPos(posView);

			std::cout << "success!!!!" << std::endl;

			//this->nodePlaced(node);
		};

	//Create a dummy module to store all state variables
	//mStates = std::make_shared<dyno::States>();

	Wt::WPointF pos = mActivePipeline == node->animationPipeline() ? SimStatePos : RenStatePos;
	//mStates->setBlockCoord(pos.x(), pos.y());

	for (auto m : modules)
	{
		addModuleWidget(m.second);
	}
}
