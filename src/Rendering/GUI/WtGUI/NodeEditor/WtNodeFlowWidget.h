#pragma once

#include <Wt/WWidget.h>
#include <Wt/WVBoxLayout.h>

#include "WtNodeFlowScene.h"
#include "WtNodeGraphicsObject.h"

class WGridLayout;

class WtNodeFlowWidget : public Wt::WPaintedWidget
{
public:
	WtNodeFlowWidget();
	~WtNodeFlowWidget();

public:
	void onMouseMove(const Wt::WMouseEvent& event);
	void onMouseWentDown(const Wt::WMouseEvent& event);
	void onMouseWentUp(const Wt::WMouseEvent& event);
	void onMouseWheel(const Wt::WMouseEvent& event);
	void zoomIn();
	void zoomOut();

protected:
	void paintEvent(Wt::WPaintDevice* paintDevice);

private:
	double mZoomFactor;
	Wt::WPointF mLastMousePos;
	Wt::WPointF mLastDelta;
	Wt::WPointF mTranlate = Wt::WPointF(0, 0);
	bool isDragging = false;
};
