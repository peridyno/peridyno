#pragma once
#include <Wt/WPaintedWidget.h>
#include <Wt/WPaintDevice.h>

class WtFlowScene;

class WtFlowView : public Wt::WPaintedWidget
{
	WtFlowView(WWidget* parent = nullptr);
	WtFlowView(WtFlowScene* scene, WWidget* parent = nullptr);

	WtFlowView(const WtFlowView&) = delete;
};