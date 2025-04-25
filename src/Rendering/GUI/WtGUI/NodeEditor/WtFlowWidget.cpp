#include "WtFlowWidget.h"

WtFlowWidget::WtFlowWidget()
{
	mZoomFactor = 1.0;
}

void WtFlowWidget::onMouseWheel(const Wt::WMouseEvent& event)
{
	if (event.wheelDelta() > 0)
	{
		zoomIn();
	}
	else
	{
		zoomOut();
	}
}

void WtFlowWidget::zoomIn()
{
	mZoomFactor *= 1.1;
	update();
}

void WtFlowWidget::zoomOut()
{
	mZoomFactor /= 1.1;
	update();
}

void WtFlowWidget::reorderNode()
{
	mZoomFactor = 1;
	reorderFlag = true;
	mTranslate = Wt::WPointF(0, 0);
	update();
}
