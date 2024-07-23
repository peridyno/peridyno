#include "WtNodeFlowWidget.h"
#include "WtFlowView.h"

WtNodeFlowWidget::WtNodeFlowWidget() :Wt::WPaintedWidget()
{
	mZoomFactor = 1.0;
	resize(600, 600);

	this->mouseWentDown().connect(this, &WtNodeFlowWidget::onMouseWentDown);
	this->mouseMoved().connect(this, &WtNodeFlowWidget::onMouseMove);
	this->mouseWentUp().connect(this, &WtNodeFlowWidget::onMouseWentUp);
	this->mouseWheel().connect(this, &WtNodeFlowWidget::onMouseWheel);
}

WtNodeFlowWidget::~WtNodeFlowWidget() {};

void WtNodeFlowWidget::onMouseWentDown(const Wt::WMouseEvent& event)
{
	//¿ªÆôÍÏ×§
	isDragging = true;
	mLastMousePos = Wt::WPointF(event.widget().x, event.widget().y);
	mLastDelta = Wt::WPointF(0, 0);
}

void WtNodeFlowWidget::onMouseMove(const Wt::WMouseEvent& event)
{
	if (isDragging)
	{
		Wt::WPointF delta = Wt::WPointF(event.dragDelta().x, event.dragDelta().y);
		mTranlate = Wt::WPointF(mTranlate.x() + delta.x() - mLastDelta.x(), mTranlate.y() + delta.y() - mLastDelta.y());
		update();
		mLastDelta = delta;
	}
}

void WtNodeFlowWidget::onMouseWentUp(const Wt::WMouseEvent& event)
{
	isDragging = false;
	mLastDelta = Wt::WPointF(0, 0);
}

void WtNodeFlowWidget::onMouseWheel(const Wt::WMouseEvent& event)
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

void WtNodeFlowWidget::zoomIn()
{
	mZoomFactor *= 1.1;
	update();
}

void WtNodeFlowWidget::zoomOut()
{
	mZoomFactor /= 1.1;
	update();
}

void WtNodeFlowWidget::paintEvent(Wt::WPaintDevice* paintDevice)
{
	Wt::WPainter painter(paintDevice);
	//WtNode node;
	//WtNodePainter::paint(&painter, node);
	painter.scale(mZoomFactor, mZoomFactor);
	painter.translate(mTranlate);
	WtNodeStyle nodeStyle;
	float diam = nodeStyle.ConnectionPointDiameter;
	Wt::WRectF boundary = Wt::WRectF(-diam, -diam, 2.0 * diam + 10, 2.0 * diam + 15);
	painter.drawRect(boundary);

	painter.save();
}