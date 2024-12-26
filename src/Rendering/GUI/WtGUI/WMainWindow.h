#pragma once

#include <Wt/WLineEdit.h>
#include <Wt/WPushButton.h>
#include <Wt/WTemplate.h>
#include <Wt/WText.h>

#include <Wt/WContainerWidget.h>
#include "WParameterDataNode.h"
#include "NodeEditor/WtFlowWidget.h"
#include "WSaveWidget.h"
#include "WLogWidget.h"

#include "NodeFactory.h"
#include <Wt/WSuggestionPopup.h>

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

	WSimulationCanvas* simCanvas() { return mSceneCanvas; }

	void createLeftPanel();

	void updateCanvas();

	void onKeyWentDown(const Wt::WKeyEvent& event);

	WtFlowWidget* getFlowWidget() {
		return mFlowWidget;
	}

	std::shared_ptr<dyno::SceneGraph> getScene();

public:
	// data models
	std::shared_ptr<WNodeDataModel>		mNodeDataModel;
	std::shared_ptr<WModuleDataModel>	mModuleDataModel;
	std::shared_ptr<WParameterDataNode> mParameterDataNode;

private:
	void initMenu(Wt::WMenu*);
	void initLeftPanel(Wt::WContainerWidget*);
	std::unique_ptr<Wt::WWidget> initNodeGraphics();
	std::unique_ptr<Wt::WWidget> initNodeTree();

	void start();
	void stop();
	void step();
	void reset();

private:

	WSimulationCanvas* mSceneCanvas;
	WtFlowWidget* mFlowWidget;

	bool	bRunFlag;
	bool	mReset;

	std::shared_ptr<dyno::SceneGraph>	mScene = nullptr;
	std::shared_ptr<dyno::Node> mActiveNode;

	Wt::WPushButton* startButton;

	Wt::WContainerWidget* widget0;
};
