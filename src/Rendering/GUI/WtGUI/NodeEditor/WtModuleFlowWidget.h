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

	void deleteModule();

	void moveModule(WtNode& n, const Wt::WPointF& newLocation);


protected:
	void paintEvent(Wt::WPaintDevice* paintDevice);

	bool checkMouseInAllRect(Wt::WPointF mousePoint);

private:
	std::shared_ptr<dyno::Node> mNode;
	
	WtModuleFlowScene* mModuleFlowScene = nullptr;
	std::map<dyno::ObjectId, WtNode*> moduleMap;

	std::shared_ptr<dyno::Module> mOutModule;

	int selectType = -1;
	int selectedNum = 0;
};