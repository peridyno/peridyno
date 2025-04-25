#pragma once

#include "WtFlowWidget.h"

class WtModuleFlowWidget : public WtFlowWidget
{
public:
	WtModuleFlowWidget(std::shared_ptr<dyno::SceneGraph> scene);
	~WtModuleFlowWidget();

	void onMouseMove(const Wt::WMouseEvent& event) override;
	void onMouseWentDown(const Wt::WMouseEvent& event) override;
	void onMouseWentUp(const Wt::WMouseEvent& event) override;
	void onKeyWentDown() override;

protected:
	void paintEvent(Wt::WPaintDevice* paintDevice);
};