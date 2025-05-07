#pragma once

#include "WtFlowScene.h"

#include "WtModuleWidget.h"

#include <Node.h>
#include <SceneGraph.h>
#include <Object.h>
//#include <QtGUI/NodeEditor/QtModuleFlowScene.cpp>

class WtModuleFlowScene : public WtFlowScene
{
public:
	WtModuleFlowScene(Wt::WPainter* painter, std::shared_ptr<dyno::Node> node);
	~WtModuleFlowScene();

public:
	void updateModuleGraphView();

	void reorderAllModules();

	void showModuleFlow(std::shared_ptr<dyno::Node> node);

	void showResetPipeline();

	void showAnimationPipeline();

	void showGraphicsPipeline();

private:
	Wt::WPainter* _painter;
	std::shared_ptr<dyno::Node> mNode;

	std::shared_ptr<dyno::Pipeline> mActivePipeline;

	//A virtual module to store all state variables
	std::shared_ptr<dyno::Module> mStates = nullptr;

	float mDx = 100.0f;
	float mDy = 50.0f;

};