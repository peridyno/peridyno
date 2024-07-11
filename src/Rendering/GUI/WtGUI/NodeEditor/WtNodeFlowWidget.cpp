#include "WtNodeFlowWidget.h"
#include "WtFlowView.h"

WtNodeFlowWidget::WtNodeFlowWidget() :Wt::WPaintedWidget()
{
	resize(200, 200);
	setWidth(100);
	setWidth(100);

	//setMouseTracking(true);

	clicked().connect(this, &WtNodeFlowWidget::handleMouseDown);
	mouseWentUp().connect(this, &WtNodeFlowWidget::handleMouseUp);
}

void WtNodeFlowWidget::handleMouseDown(Wt::WMouseEvent event)
{
	lastPosition_.setX(event.widget().x);
	lastPosition_.setY(event.widget().y);

	isDragging_ = true;
}

void WtNodeFlowWidget::handleMouseUp(Wt::WMouseEvent event)
{
	if (isDragging_)
	{
		isDragging_ = false;
	}
}

void WtNodeFlowWidget::mouseMove(Wt::WMouseEvent event)
{
	if (isDragging_)
	{
		Wt::WPointF currentPostion;
		currentPostion.setX(event.widget().x);
		currentPostion.setY(event.widget().y);
		//Wt::WPointF delta = currentPostion - lastPosition_;
		update();
	}
}

void WtNodeFlowWidget::paintEvent(Wt::WPaintDevice* paintDevice)
{
	Wt::WPainter painter(paintDevice);
	painter.setPen(Wt::WPen(Wt::WColor("black")));
	painter.setBrush(Wt::WBrush(Wt::WColor("red")));
	Wt::WPainterPath filledEllipsePath = Wt::WPainterPath();
	filledEllipsePath.addEllipse(0, 0, 100, 100);
	filledEllipsePath.closeSubPath();
	painter.drawPath(filledEllipsePath);
}

WtNodeFlowWidget::~WtNodeFlowWidget() {};