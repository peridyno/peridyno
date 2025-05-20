#pragma once

#include <Wt/WBorderLayout.h>
#include <Wt/WContainerWidget.h>
#include <Wt/WPanel.h>
#include <Wt/WTabWidget.h>

#include "WModuleGraphics.h"
#include "WNodeGraphics.h"
#include "WPythonWidget.h"
#include "WSimulationControl.h"
#include "WSceneDataModel.h"
#include "WParameterDataNode.h"

#include "NodeEditor/WtNodeFlowWidget.h"
#include "NodeEditor/WtModuleFlowWidget.h"

namespace dyno
{
	class SceneGraph;
	class SceneGraphFactory;
	class Node;
};

enum AddNodeType { NodeType, ModuleType };

class WMainWindow : public Wt::WContainerWidget
{
public:
	WMainWindow();
	~WMainWindow();

	void setScene(std::shared_ptr<dyno::SceneGraph> scene);

	std::shared_ptr<dyno::SceneGraph> getScene();

	void createRightPanel();

	void updateCanvas();

	void updateNodeGraphics();

	void onKeyWentDown(const Wt::WKeyEvent& event);

	WSimulationCanvas* simCanvas() { return mSceneCanvas; }

	WtNodeFlowWidget* getFlowWidget() { return mNodeFlowWidget; }

public:
	// data models
	std::shared_ptr<WNodeDataModel>		mNodeDataModel;
	std::shared_ptr<WModuleDataModel>	mModuleDataModel;
	std::shared_ptr<WParameterDataNode> mParameterDataNode;

private:
	void initNavigationBar(Wt::WBorderLayout*);
	void initCenterContainer(Wt::WBorderLayout*);
	void initRightPanel(Wt::WContainerWidget*);
	void initAddNodePanel(Wt::WPanel* parent, AddNodeType addNodeType = NodeType);
	void initPipelinePanel(Wt::WPanel* parent);

	std::unique_ptr<Wt::WWidget> initNodeGraphics();
	std::unique_ptr<Wt::WWidget> initPython();
	std::unique_ptr<Wt::WWidget> initSample();
	std::unique_ptr<Wt::WWidget> initSave();
	std::unique_ptr<Wt::WWidget> initLog();
	std::unique_ptr<Wt::WWidget> initModuleGraphics();

private:
	int viewportHeight;
	int viewportWidth;

	std::shared_ptr<dyno::SceneGraph>	mScene = nullptr;
	std::shared_ptr<dyno::Node> mActiveNode = nullptr;

	WSimulationCanvas* mSceneCanvas;
	WSimulationControl* controlContainer;
	WtNodeFlowWidget* mNodeFlowWidget;
	WtModuleFlowWidget* mModuleFlowWidget;
	WPythonWidget* pythonWidget = new WPythonWidget();
	std::unique_ptr<WNodeGraphics> nodeGraphicsWidget;
	std::unique_ptr<WModuleGraphics> moduleGraphicsWidget;

	Wt::WContainerWidget* rightWidget;

	Wt::WTabWidget* tab;

	int Initial_x = 0;
	int Initial_y = 0;
};
