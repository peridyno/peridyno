#pragma once

#include "WtFlowWidget.h"
#include "WtModuleFlowScene.h"

class WtModuleFlowWidget : public WtFlowWidget
{
public:
	WtModuleFlowWidget(std::shared_ptr<dyno::SceneGraph> scene);
	~WtModuleFlowWidget();

	void onMouseMove(const Wt::WMouseEvent& event) override;
	void onMouseWentDown(const Wt::WMouseEvent& event) override;
	void onMouseWentUp(const Wt::WMouseEvent& event) override;
	void onKeyWentDown() override;

	void setNode(std::shared_ptr<dyno::Node> node);

protected:
	void paintEvent(Wt::WPaintDevice* paintDevice);

private:
	std::shared_ptr<dyno::Node> mNode;
	WtModuleFlowScene* mModuleFlowScene = nullptr;
};