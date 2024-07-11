#pragma once

#include <Wt/WPaintDevice.h>
#include <Wt/WPaintedWidget.h>
#include <Wt/WPainter.h>

class WtFlowScene : public Wt::WPaintedWidget
{
	WtFlowScene(WObject* parent = nullptr);
	~WtFlowScene();
};