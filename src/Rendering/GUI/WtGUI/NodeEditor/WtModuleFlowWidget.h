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

	void addModule(std::shared_ptr<dyno::Module> new_module);

	void deleteModule(std::shared_ptr<dyno::Module> delete_module);

	void moveModule(WtNode& n, const Wt::WPointF& newLocation);

	void showResetPipeline();

	void showAnimationPipeline();

	void showGraphicsPipeline();

protected:
	void paintEvent(Wt::WPaintDevice* paintDevice);

	bool checkMouseInAllRect(Wt::WPointF mousePoint);

	void disconnect(std::shared_ptr<Module> exportModule, std::shared_ptr<Module> inportModule, connectionPointData inPoint, connectionPointData outPoint, WtNode* inWtNode, WtNode* outWtNode);

private:
	std::shared_ptr<dyno::Node> mNode;

	WtModuleFlowScene* mModuleFlowScene = nullptr;
	std::map<dyno::ObjectId, WtNode*> moduleMap;

	WtNode* connectionOutNode;

	std::shared_ptr<dyno::Module> mOutModule;

	PipelineType pipelineType = PipelineType::Animation;

	int selectType = -1;
	int selectedNum = 0;

	std::vector<sceneConnection> nodeConnections;
};