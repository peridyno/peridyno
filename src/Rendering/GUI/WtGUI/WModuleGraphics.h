#pragma once

// wt
#include <Wt/WContainerWidget.h>
#include <Wt/WPanel.h>
#include <Wt/WVBoxLayout.h>

class WModuleGraphics : public Wt::WContainerWidget
{
public:
	WModuleGraphics();
	~WModuleGraphics();

public:
	Wt::WVBoxLayout* layout;
	Wt::WPanel* addPanel;
	Wt::WPanel* pipelinePanel;
	Wt::WPanel* modulePanel;
};