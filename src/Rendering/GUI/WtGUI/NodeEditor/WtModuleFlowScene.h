#pragma once

#include "WtFlowScene.h"

#include "WtModuleWidget.h"

#include <Node.h>
#include <SceneGraph.h>
//#include <QtGUI/NodeEditor/QtModuleFlowScene.cpp>

class WtModuleFlowScene : public WtFlowScene
{
public:
	WtModuleFlowScene(Wt::WPainter* painter, std::shared_ptr<dyno::Node> node, std::shared_ptr<dyno::SceneGraph> scene);
	~WtModuleFlowScene();

public:
	void addModule(WtNode& n);

	void showModuleFlow(std::shared_ptr<dyno::Node> node);

private:
	Wt::WPainter* _painter;
	std::shared_ptr<dyno::Node> mNode;

	std::shared_ptr<dyno::Pipeline> mActivePipeline;
	std::shared_ptr<dyno::SceneGraph> mScene = nullptr;



};