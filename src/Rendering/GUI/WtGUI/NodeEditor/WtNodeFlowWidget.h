#pragma once

#include <Wt/WWidget.h>
#include <Wt/WVBoxLayout.h>

#include "WtFlowWidget.h"
#include "WtNodeFlowScene.h"
#include "WtNodeGraphicsObject.h"
#include "WtFlowNodeData.h"
#include "WtInteraction.h"

class WtNodeFlowWidget : public WtFlowWidget
{
public:
	WtNodeFlowWidget(std::shared_ptr<dyno::SceneGraph> scene);
	~WtNodeFlowWidget();

public:
	void onMouseMove(const Wt::WMouseEvent& event) override;
	void onMouseWentDown(const Wt::WMouseEvent& event) override;
	void onMouseWentUp(const Wt::WMouseEvent& event) override;
	void onKeyWentDown() override;

	void deleteNode(WtNode& n);

	void disconnectionsFromNode(WtNode& node);

	void moveNode(WtNode& n, const Wt::WPointF& newLocaton);

	void enableRendering(WtNode& n, bool checked);

	void enablePhysics(WtNode& n, bool checked);

	void setSelectNode(std::shared_ptr<dyno::Node> node);

protected:
	void paintEvent(Wt::WPaintDevice* paintDevice);

	bool checkMouseInAllRect(Wt::WPointF mousePoint);

	bool checkMouseInHotKey0(Wt::WPointF mousePoint, WtFlowNodeData nodeData);

	bool checkMouseInHotKey1(Wt::WPointF mousePoint, WtFlowNodeData nodeData);

	void disconnect(std::shared_ptr<Node> exportNode, std::shared_ptr<Node> inportNode, connectionPointData inPoint, connectionPointData outPoint, WtNode* inWtNode, WtNode* outWtNode);

protected:
	WtNodeFlowScene* mNodeFlowScene = nullptr;
	std::map<dyno::ObjectId, WtNode*> nodeMap;
	std::map<dyno::ObjectId, std::shared_ptr<dyno::Node>> allNodeMap;
	WtNode* connectionOutNode;
	bool isConnect = false;

	std::shared_ptr<Node> mOutNode;
};
