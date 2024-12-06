#pragma once

#include <Wt/WWidget.h>
#include <Wt/WVBoxLayout.h>

#include "WtNodeFlowScene.h"
#include "WtNodeGraphicsObject.h"
#include "WtFlowNodeData.h"
#include "WMainWindow.h"
#include "WtInteraction.h"

class WGridLayout;
class WMainWindow;

enum PortState { in, out };

class WtFlowWidget : public Wt::WPaintedWidget
{
public:
	//WtFlowWidget(std::shared_ptr<dyno::SceneGraph> scene);
	WtFlowWidget(std::shared_ptr<dyno::SceneGraph> scene, WMainWindow* mainWindow);
	~WtFlowWidget();

public:
	void onMouseMove(const Wt::WMouseEvent& event);
	void onMouseWentDown(const Wt::WMouseEvent& event);
	void onMouseWentUp(const Wt::WMouseEvent& event);
	void onMouseWheel(const Wt::WMouseEvent& event);
	void zoomIn();
	void zoomOut();

	void onKeyWentDown();

	void deleteNode(WtNode& n);

	void moveNode(WtNode& n, const Wt::WPointF& newLocaton);

	void enableRendering(WtNode& n, bool checked);

	void enablePhysics(WtNode& n, bool checked);

	void updateForAddNode();

	void reorderNode();

protected:
	void paintEvent(Wt::WPaintDevice* paintDevice);

private:
	bool checkMouseInNodeRect(Wt::WPointF mousePoint, WtFlowNodeData nodeData);

	bool checkMouseInHotKey0(Wt::WPointF mousePoint, WtFlowNodeData nodeData);

	bool checkMouseInHotKey1(Wt::WPointF mousePoint, WtFlowNodeData nodeData);

	bool checkMouseInPoints(Wt::WPointF mousePoint, WtFlowNodeData nodeData, PortState portState);

	Wt::WPainterPath cubicPath(Wt::WPointF source, Wt::WPointF sink);
	std::pair<Wt::WPointF, Wt::WPointF> pointsC1C2(Wt::WPointF source, Wt::WPointF sink);
	void drawSketchLine(Wt::WPainter* painter, Wt::WPointF source, Wt::WPointF sink);

private:
	double mZoomFactor;
	Wt::WPointF mLastMousePos;
	Wt::WPointF mLastDelta;
	Wt::WPointF mTranslate = Wt::WPointF(0, 0);
	Wt::WPointF mTranslateNode = Wt::WPointF(0, 0);

	bool isDragging = false;
	bool isSelected = false;
	int selectedNum = 0;
	bool canMoveNode = false;

	bool reorderFlag = true;

	bool mEditingEnabled = true;

	bool drawLineFlag = false;
	Wt::WPointF sourcePoint;
	Wt::WPointF sinkPoint;

	connectionPointData outPoint;
	connectionPointData inPoint;

	WMainWindow* mMainWindow = nullptr;

	WtNodeFlowScene* node_scene = nullptr;
	std::shared_ptr<dyno::SceneGraph> mScene;
	std::map<dyno::ObjectId, WtNode*> nodeMap;
	WtNode* connectionOutNode;

	Wt::WPointF mMousePoint = Wt::WPointF(0, 0);
};
