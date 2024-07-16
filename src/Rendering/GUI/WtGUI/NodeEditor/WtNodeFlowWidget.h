#pragma once

#include <Wt/WWidget.h>
#include <Wt/WVBoxLayout.h>

#include "WtNodeFlowScene.h"
#include "WtNodeGraphicsObject.h"

class WGridLayout;

class WtNodeFlowWidget : public Wt::WPaintedWidget
{
public:
	WtNodeFlowWidget();
	~WtNodeFlowWidget();
protected:
	void paintEvent(Wt::WPaintDevice* paintDevice);
};
