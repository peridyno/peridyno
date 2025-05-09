#pragma once

#include <Wt/WWidget.h>
#include <Wt/WVBoxLayout.h>

#include "WtFlowWidget.h"
#include "WtNodeFlowScene.h"
#include "WtNodeGraphicsObject.h"
#include "WtFlowNodeData.h"
#include "WtInteraction.h"

//enum PortState { in, out };
//
//struct sceneConnection {
//	std::shared_ptr<Node> exportNode;
//	std::shared_ptr<Node> inportNode;
//	connectionPointData inPoint;
//	connectionPointData outPoint;
//};

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

	Wt::WPointF WtNodeFlowWidget::getPortPosition(Wt::WPointF origin, connectionPointData portData);

	void disconnect(std::shared_ptr<Node> exportNode, std::shared_ptr<Node> inportNode, connectionPointData inPoint, connectionPointData outPoint, WtNode* inWtNode, WtNode* outWtNode);

protected:
	WtNodeFlowScene* node_scene = nullptr;
	std::map<dyno::ObjectId, WtNode*> nodeMap;
	WtNode* connectionOutNode;

	std::shared_ptr<Node> mOutNode;
	std::vector<sceneConnection> sceneConnections;
};
