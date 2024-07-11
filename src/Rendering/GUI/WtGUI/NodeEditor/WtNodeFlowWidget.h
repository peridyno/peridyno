#pragma once

#include <Wt/WWidget.h>
#include <Wt/WVBoxLayout.h>

#include "WtNodeFlowScene.h"

class WGridLayout;

class WtNodeFlowWidget : public Wt::WPaintedWidget
{
public:
	WtNodeFlowWidget();
	~WtNodeFlowWidget();

private:
	bool isDragging_ = false;
	Wt::WPointF lastPosition_;

	//void setMouseTracking(bool);
	void handleMouseDown(Wt::WMouseEvent event);
	void handleMouseUp(Wt::WMouseEvent event);
	void mouseMove(Wt::WMouseEvent event);

protected:
	void paintEvent(Wt::WPaintDevice* paintDevice);
};
