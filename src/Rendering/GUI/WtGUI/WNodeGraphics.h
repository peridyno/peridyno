#pragma once

// wt
#include <Wt/WContainerWidget.h>
#include <Wt/WVBoxLayout.h>
#include <Wt/WHBoxLayout.h>
#include <Wt/WPanel.h>
#include <Wt/WSuggestionPopup.h>
#include <Wt/WLineEdit.h>
#include <Wt/WPushButton.h>

// dyno
#include <SceneGraph.h>
#include <SceneGraphFactory.h>
#include <NodeFactory.h>

#include "NodeEditor/WtFlowWidget.h"

class WNodeGraphics : public Wt::WContainerWidget
{

public:
	WNodeGraphics();
	~WNodeGraphics();

public:
	void setSceneGraph(std::shared_ptr<dyno::SceneGraph> Scenes) { mScene = Scenes; }

public:
	Wt::WPanel* addPanel;
	Wt::WPanel* nodePanel;
	Wt::WPanel* parameterPanel;

private:
	void initAddNodePanel(Wt::WPanel* panel);

private:
	std::shared_ptr<dyno::SceneGraph>	mScene = nullptr;

	
};