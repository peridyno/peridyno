#pragma once

#include "WtFlowScene.h"

#include "WtModuleWidget.h"

#include <Node.h>
#include <SceneGraph.h>
#include <Object.h>
#include "WtFlowWidget.h"
//#include <QtGUI/NodeEditor/QtModuleFlowScene.cpp>

enum class PipelineType {Reset, Animation, Graphics};

class WtModuleFlowScene : public WtFlowScene
{
public:
	WtModuleFlowScene(Wt::WPainter* painter, std::shared_ptr<dyno::Node> node, PipelineType pipelineType);
	~WtModuleFlowScene();

public:
	void enableEditing();

	void disableEditing();

	void updateModuleGraphView();

	void reorderAllModules();

	void showModuleFlow(std::shared_ptr<dyno::Node> node);

	void addModule(std::shared_ptr<dyno::Module> new_module);

	void deleteModule(std::shared_ptr<dyno::Module> delete_module);

	void showResetPipeline();

	void showAnimationPipeline();

	void showGraphicsPipeline();

	std::vector<connectionData> getConnections() { return nodeConnections; }

private:
	void addConnection(std::shared_ptr<dyno::Module> exportModule, std::shared_ptr<dyno::Module> inportModule);

private:
	Wt::WPainter* _painter;
	std::shared_ptr<dyno::Node> mNode;

	std::shared_ptr<dyno::Pipeline> mActivePipeline;

	//A virtual module to store all state variables
	std::shared_ptr<dyno::Module> mStates = nullptr;

	PipelineType mPipelineType;

	bool mEditingEnabled = true;

	float mDx = 100.0f;
	float mDy = 50.0f;

	std::vector<connectionData> nodeConnections;
};