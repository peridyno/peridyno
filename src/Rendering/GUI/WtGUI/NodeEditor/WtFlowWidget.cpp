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

bool WtFlowWidget::checkMouseInRect(Wt::WPointF mousePoint, WtFlowNodeData nodeData)
{
	Wt::WPointF bottomRight = Wt::WPointF(nodeData.getNodeBoundingRect().bottomRight().x() + nodeData.getNodeOrigin().x()
		, nodeData.getNodeBoundingRect().bottomRight().y() + nodeData.getNodeOrigin().y());

	Wt::WPointF absTopLeft = Wt::WPointF((nodeData.getNodeOrigin().x() + mTranslate.x() - 10) * mZoomFactor, (nodeData.getNodeOrigin().y() + mTranslate.y() - 10) * mZoomFactor);
	Wt::WPointF absBottomRight = Wt::WPointF((bottomRight.x() + mTranslate.x() + 10) * mZoomFactor, (bottomRight.y() + mTranslate.y() + 10) * mZoomFactor);

	Wt::WRectF absRect = Wt::WRectF(absTopLeft, absBottomRight);

	return absRect.contains(mousePoint);
}

bool WtFlowWidget::checkMouseInPoints(Wt::WPointF mousePoint, WtFlowNodeData nodeData, PortState portState)
{
	auto pointsData = nodeData.getPointsData();
	Wt::WPointF origin = nodeData.getNodeOrigin();
	Wt::WPointF	trueMouse = Wt::WPointF(mousePoint.x() / mZoomFactor - mTranslate.x(), mousePoint.y() / mZoomFactor - mTranslate.y());

	for (connectionPointData pointData : pointsData)
	{
		if (pointData.portShape == PortShape::Bullet)
		{
			Wt::WPointF topLeft = Wt::WPointF(pointData.diamond_out[3].x() + origin.x(), pointData.diamond_out[2].y() + origin.y());
			Wt::WPointF bottomRight = Wt::WPointF(pointData.diamond_out[1].x() + origin.x(), pointData.diamond_out[0].y() + origin.y());
			Wt::WRectF diamondBoundingRect = Wt::WRectF(topLeft, bottomRight);
			if (diamondBoundingRect.contains(trueMouse))
			{
				sourcePoint = Wt::WPointF((topLeft.x() + bottomRight.x()) / 2, (topLeft.y() + bottomRight.y()) / 2);
				if (portState == PortState::out)
				{
					outPoint = pointData;
				}
				if (portState == PortState::in)
				{
					inPoint = pointData;
				}
				return true;
			}
		}
		else if (pointData.portShape == PortShape::Diamond)
		{
			Wt::WPointF topLeft = Wt::WPointF(pointData.diamond[3].x() + origin.x(), pointData.diamond[2].y() + origin.y());
			Wt::WPointF bottomRight = Wt::WPointF(pointData.diamond[1].x() + origin.x(), pointData.diamond[0].y() + origin.y());
			Wt::WRectF diamondBoundingRect = Wt::WRectF(topLeft, bottomRight);
			if (diamondBoundingRect.contains(trueMouse))
			{
				sourcePoint = Wt::WPointF((topLeft.x() + bottomRight.x()) / 2, (topLeft.y() + bottomRight.y()) / 2);
				if (portState == PortState::out)
				{
					outPoint = pointData;
				}
				if (portState == PortState::in)
				{
					inPoint = pointData;
				}
				return true;
			}
		}
		else if (pointData.portShape == PortShape::Point)
		{
			auto rectTopLeft = pointData.pointRect.topLeft();
			auto rectBottomRight = pointData.pointRect.bottomRight();
			Wt::WPointF topLeft = Wt::WPointF(rectTopLeft.x() + origin.x(), rectTopLeft.y() + origin.y());
			Wt::WPointF bottomRight = Wt::WPointF(rectBottomRight.x() + origin.x(), rectBottomRight.y() + origin.y());
			Wt::WRectF diamondBoundingRect = Wt::WRectF(topLeft, bottomRight);
			if (diamondBoundingRect.contains(trueMouse))
			{
				sourcePoint = Wt::WPointF((topLeft.x() + bottomRight.x()) / 2, (topLeft.y() + bottomRight.y()) / 2);
				if (portState == PortState::out)
				{
					outPoint = pointData;
				}
				if (portState == PortState::in)
				{
					inPoint = pointData;
				}
				return true;
			}
		}
	}
	return false;
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

Wt::WPointF WtFlowWidget::getPortPosition(Wt::WPointF origin, connectionPointData portData)
{
	if (portData.portShape == PortShape::Bullet)
	{
		Wt::WPointF topLeft = Wt::WPointF(portData.diamond_out[3].x() + origin.x(), portData.diamond_out[2].y() + origin.y());
		Wt::WPointF bottomRight = Wt::WPointF(portData.diamond_out[1].x() + origin.x(), portData.diamond_out[0].y() + origin.y());
		Wt::WRectF diamondBoundingRect = Wt::WRectF(topLeft, bottomRight);
		return Wt::WPointF((topLeft.x() + bottomRight.x()) / 2, (topLeft.y() + bottomRight.y()) / 2);
	}
	else if (portData.portShape == PortShape::Diamond)
	{
		Wt::WPointF topLeft = Wt::WPointF(portData.diamond[3].x() + origin.x(), portData.diamond[2].y() + origin.y());
		Wt::WPointF bottomRight = Wt::WPointF(portData.diamond[1].x() + origin.x(), portData.diamond[0].y() + origin.y());
		Wt::WRectF diamondBoundingRect = Wt::WRectF(topLeft, bottomRight);
		return Wt::WPointF((topLeft.x() + bottomRight.x()) / 2, (topLeft.y() + bottomRight.y()) / 2);
	}
	else if (portData.portShape == PortShape::Point)
	{
		auto rectTopLeft = portData.pointRect.topLeft();
		auto rectBottomRight = portData.pointRect.bottomRight();
		Wt::WPointF topLeft = Wt::WPointF(rectTopLeft.x() + origin.x(), rectTopLeft.y() + origin.y());
		Wt::WPointF bottomRight = Wt::WPointF(rectBottomRight.x() + origin.x(), rectBottomRight.y() + origin.y());
		Wt::WRectF diamondBoundingRect = Wt::WRectF(topLeft, bottomRight);
		return Wt::WPointF((topLeft.x() + bottomRight.x()) / 2, (topLeft.y() + bottomRight.y()) / 2);
	}
}
