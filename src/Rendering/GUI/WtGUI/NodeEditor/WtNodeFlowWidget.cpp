#include "WtNodeFlowWidget.h"
#include "WtFlowView.h"

WtNodeFlowWidget::WtNodeFlowWidget() :Wt::WPaintedWidget()
{
	resize(200, 200);
}

WtNodeFlowWidget::~WtNodeFlowWidget() {};

void WtNodeFlowWidget::paintEvent(Wt::WPaintDevice* paintDevice)
{
	Wt::WPainter painter(paintDevice);
	painter.setPen(Wt::WPen(Wt::WColor("black")));
	painter.setBrush(Wt::WBrush(Wt::WColor("red")));
	WtNode node;
	WtNodePainter::paint(&painter, node);

}

