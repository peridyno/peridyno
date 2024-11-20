#pragma once

#include <Wt/WWidget.h>
#include <Wt/WVBoxLayout.h>

#include "WtNodeFlowScene.h"
#include "WtNodeGraphicsObject.h"
#include "WtFlowNodeData.h"

class WGridLayout;

class WtFlowWidget : public Wt::WPaintedWidget
{
public:
	WtFlowWidget(std::shared_ptr<dyno::SceneGraph> scene);
	~WtFlowWidget();

public:
	void onMouseMove(const Wt::WMouseEvent& event);
	void onMouseWentDown(const Wt::WMouseEvent& event);
	void onMouseWentUp(const Wt::WMouseEvent& event);
	void onMouseWheel(const Wt::WMouseEvent& event);
	void onKeyWentDown(const Wt::WKeyEvent& event);
	void zoomIn();
	void zoomOut();

	void addNode(WtNode& n);

	void deleteNode(WtNode& n);

	void moveNode(WtNode& n, const Wt::WPointF& newLocaton);

	void enableRendering(WtNode& n, bool checked);

	void enablePhysics(WtNode& n, bool checked);

protected:
	void paintEvent(Wt::WPaintDevice* paintDevice);

private:
	bool checkMouseInNodeRect(Wt::WPointF mousePoint, WtFlowNodeData nodeData);

	bool checkMouseInHotKey0(Wt::WPointF mousePoint, WtFlowNodeData nodeData);

	bool checkMouseInHotKey1(Wt::WPointF mousePoint, WtFlowNodeData nodeData);

private:
	double mZoomFactor;
	Wt::WPointF mLastMousePos;
	Wt::WPointF mLastDelta;
	Wt::WPointF mTranslate = Wt::WPointF(0, 0);
	Wt::WPointF mTranslateNode = Wt::WPointF(0, 0);

	bool isDragging = false;
	bool isSelected;
	int selectedNum;
	bool canMoveNode = false;

	bool reorderFlag = true;

	bool mEditingEnabled = true;



	WtNodeFlowScene* node_scene = nullptr;
	std::shared_ptr<dyno::SceneGraph> mScene;
	std::map<dyno::ObjectId, WtNode*> nodeMap;
};
