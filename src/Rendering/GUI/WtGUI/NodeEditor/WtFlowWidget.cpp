#include "WtFlowWidget.h"

#include "WtNodeStyle.h"
#include <Wt/WPen.h>

WtFlowWidget::WtFlowWidget(std::shared_ptr<dyno::SceneGraph> scene)
	: mScene(scene)
	, mZoomFactor(1.0)
{
	this->mouseWheel().connect(this, &WtFlowWidget::onMouseWheel);
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

void WtFlowWidget::updateAll()
{
	update();
	mScene->setFrameNumber(0);
	mScene->reset();
	_updateCanvas.emit();
}

Wt::WPainterPath WtFlowWidget::cubicPath(Wt::WPointF source, Wt::WPointF sink)
{
	auto c1c2 = pointsC1C2(source, sink);

	//cubic spline
	Wt::WPainterPath cubic(source);

	cubic.cubicTo(c1c2.first, c1c2.second, sink);

	return cubic;
}

std::pair<Wt::WPointF, Wt::WPointF> WtFlowWidget::pointsC1C2(Wt::WPointF source, Wt::WPointF sink)
{
	const double defaultOffset = 200;

	double xDistance = sink.x() - source.x();

	double horizontalOffset = std::min(defaultOffset, std::abs(xDistance));

	double verticalOffset = 0;

	double ratioX = 0.5;

	if (xDistance <= 0)
	{
		double yDistance = sink.y() - source.y();

		double vector = yDistance < 0 ? -1.0 : 1.0;

		verticalOffset = std::min(defaultOffset, std::abs(yDistance)) * vector;

		ratioX = 1.0;
	}

	horizontalOffset *= ratioX;

	Wt::WPointF c1(source.x() + horizontalOffset, source.y() + verticalOffset);
	Wt::WPointF c2(sink.x() - horizontalOffset, sink.y() - verticalOffset);

	return std::make_pair(c1, c2);
}

void WtFlowWidget::drawSketchLine(Wt::WPainter* painter, Wt::WPointF source, Wt::WPointF sink)
{
	auto const& connectionStyle = WtStyleCollection::connectionStyle();

	Wt::WPen p;
	p.setWidth(connectionStyle.constructionLineWidth());
	p.setColor(connectionStyle.constructionColor());
	p.setStyle(Wt::PenStyle::DashLine);

	painter->setPen(p);
	painter->setBrush(Wt::BrushStyle::None);

	auto cubic = cubicPath(source, sink);

	painter->drawPath(cubic);
}