#pragma once

#include <Wt/WLineEdit.h>
#include <Wt/WPushButton.h>
#include <Wt/WTemplate.h>
#include <Wt/WText.h>
#include <Wt/WSuggestionPopup.h>
#include <Wt/WApplication.h>
#include <Wt/WVBoxLayout.h>
#include <Wt/WHBoxLayout.h>
#include <Wt/WBorderLayout.h>
#include <Wt/WPushButton.h>
#include <Wt/WPanel.h>
#include <Wt/WMenu.h>
#include <Wt/WPopupMenu.h>
#include <Wt/WNavigationBar.h>
#include <Wt/WTreeView.h>
#include <Wt/WTableView.h>
#include <Wt/WText.h>
#include <Wt/WTable.h>
#include <Wt/WColorPicker.h>
#include <Wt/WLogger.h>
#include <Wt/WTabWidget.h>
#include <Wt/WTextArea.h>
#include <Wt/WApplication.h>
#include <Wt/WEnvironment.h>

#include <fstream>
#include <filesystem>

#include <SceneGraph.h>
#include <SceneGraphFactory.h>


#include <Wt/WContainerWidget.h>
#include "WParameterDataNode.h"
#include "NodeEditor/WtFlowWidget.h"
#include "WNodeGraphics.h"
#include "WModuleGraphics.h"
#include "WSaveWidget.h"
#include "WLogWidget.h"
#include "NodeFactory.h"
#include "WPythonWidget.h"
#include "WSceneDataModel.h"
#include "WSimulationCanvas.h"
#include "WSimulationControl.h"
#include "WSampleWidget.h"
#include "WRenderParamsWidget.h"
#include "WPythonWidget.h"
#include "WParameterDataNode.h"


namespace dyno
{
	class SceneGraph;
	class SceneGraphFactory;
	class Node;
};

class WNodeDataModel;
class WModuleDataModel;
class WParameterDataNode;
class WSimulationCanvas;
class WPushButton;
class WtFlowWidget;

class WMainWindow : public Wt::WContainerWidget
{
public:
	WMainWindow();
	~WMainWindow();

	void setScene(std::shared_ptr<dyno::SceneGraph> scene);

	std::shared_ptr<dyno::SceneGraph> getScene();

	void createRightPanel();

	void updateCanvas();

	void onKeyWentDown(const Wt::WKeyEvent& event);

	WSimulationCanvas* simCanvas() { return mSceneCanvas; }

	WtFlowWidget* getFlowWidget() { return mFlowWidget; }

public:
	// data models
	std::shared_ptr<WNodeDataModel>		mNodeDataModel;
	std::shared_ptr<WModuleDataModel>	mModuleDataModel;
	std::shared_ptr<WParameterDataNode> mParameterDataNode;

private:
	void initNavigationBar(Wt::WBorderLayout*);
	void initCenterContainer(Wt::WBorderLayout*);
	void initRightPanel(Wt::WContainerWidget*);
	void initAddNodePanel(Wt::WPanel* parent);


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
	std::shared_ptr<dyno::Node> mActiveNode;

	WSimulationCanvas* mSceneCanvas;
	WSimulationControl* controlContainer;
	WtFlowWidget* mFlowWidget;
	WPythonWidget* pythonWidget;

	Wt::WContainerWidget* rightWidget;

	Wt::WTabWidget* tab;

	int Initial_x = 0;
	int Initial_y = 0;
};
