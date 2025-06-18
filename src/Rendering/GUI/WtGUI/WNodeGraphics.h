#pragma once

// wt
#include <Wt/WContainerWidget.h>
#include <Wt/WPanel.h>
#include <Wt/WVBoxLayout.h>



class WNodeGraphics : public Wt::WContainerWidget
{

public:
	WNodeGraphics();
	~WNodeGraphics();

public:
	Wt::WVBoxLayout* layout;
	Wt::WPanel* addPanel;
	Wt::WPanel* nodePanel;
};