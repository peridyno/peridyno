#include "WtFlowWidget.h"

WtFlowWidget::WtFlowWidget(std::shared_ptr<dyno::SceneGraph> scene, WMainWindow* mainWindow) :Wt::WPaintedWidget()
{
	mZoomFactor = 1.0;
	mScene = scene;

	mMainWindow = mainWindow;

	resize(900, 1200);

	std::cout << "WtFlowWidget" << std::endl;

	setPreferredMethod(Wt::RenderMethod::HtmlCanvas);

	this->mouseWentDown().connect(this, &WtFlowWidget::onMouseWentDown);
	this->mouseMoved().connect(this, &WtFlowWidget::onMouseMove);
	this->mouseWentUp().connect(this, &WtFlowWidget::onMouseWentUp);
	this->mouseWheel().connect(this, &WtFlowWidget::onMouseWheel);
}

WtFlowWidget::~WtFlowWidget() {};

void WtFlowWidget::onMouseWentDown(const Wt::WMouseEvent& event)
{
	isDragging = true;
	mLastMousePos = Wt::WPointF(event.widget().x, event.widget().y);
	mLastDelta = Wt::WPointF(0, 0);
	if (isSelected)
	{
		auto origin = nodeMap[selectedNum]->flowNodeData().getNodeOrigin();
		mTranslateNode = Wt::WPointF(origin.x(), origin.y());
	}
}

void WtFlowWidget::onMouseMove(const Wt::WMouseEvent& event)
{
	if (isDragging && !isSelected)
	{
		Wt::WPointF delta = Wt::WPointF(event.dragDelta().x, event.dragDelta().y);
		mTranslate = Wt::WPointF(mTranslate.x() + delta.x() - mLastDelta.x(), mTranslate.y() + delta.y() - mLastDelta.y());
		update();
		mLastDelta = delta;
	}
	else if (isDragging && isSelected)
	{
		Wt::WPointF delta = Wt::WPointF(event.dragDelta().x, event.dragDelta().y);
		mTranslateNode = Wt::WPointF(mTranslateNode.x() + delta.x() - mLastDelta.x(), mTranslateNode.y() + delta.y() - mLastDelta.y());
		update();
		mLastDelta = delta;
	}
	else
	{
		for (auto it = mScene->begin(); it != mScene->end(); it++)
		{
			auto m = it.get();
			auto node = nodeMap[m->objectId()];
			auto nodeData = node->flowNodeData();
			auto mousePoint = Wt::WPointF(event.widget().x, event.widget().y);
			if (checkMouseInNodeRect(mousePoint, nodeData))
			{
				isSelected = true;
				selectedNum = m->objectId();
				canMoveNode = true;
				update();
				break;
			}
			else
			{
				isSelected = false;
				selectedNum = 0;
				canMoveNode = false;
				//update();
			}
		}
	}
}

void WtFlowWidget::onMouseWentUp(const Wt::WMouseEvent& event)
{
	isDragging = false;
	mLastDelta = Wt::WPointF(0, 0);
	mTranslateNode = Wt::WPointF(0, 0);
	if (isSelected)
	{
		auto node = nodeMap[selectedNum];
		auto nodeData = node->flowNodeData();
		Wt::WPointF mousePoint = Wt::WPointF(event.widget().x, event.widget().y);
		if (checkMouseInHotKey0(mousePoint, nodeData))
		{
			auto nodeWidget = dynamic_cast<WtNodeWidget*>(node->nodeDataModel());
			auto m = nodeWidget->getNode();
			if (m->isVisible())
			{
				enableRendering(*node, false);
			}
			else
			{
				enableRendering(*node, true);
			}
			mMainWindow->updateCanvas();
			update();
		}

		if (checkMouseInHotKey1(mousePoint, nodeData))
		{
			auto nodeWidget = dynamic_cast<WtNodeWidget*>(node->nodeDataModel());
			auto m = nodeWidget->getNode();
			if (m->isActive())
			{
				enablePhysics(*node, false);
			}
			else
			{
				enablePhysics(*node, true);
			}
			mMainWindow->updateCanvas();
			update();
		}
	}
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

void WtFlowWidget::onKeyWentDown()
{
	if (isSelected)
	{
		auto node = nodeMap[selectedNum];
		deleteNode(*node);
		isSelected = false;
		selectedNum = 0;
		updateForAddNode();
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

void WtFlowWidget::paintEvent(Wt::WPaintDevice* paintDevice)
{
	Wt::WPainter painter(paintDevice);
	painter.scale(mZoomFactor, mZoomFactor);
	painter.translate(mTranslate);

	if (reorderFlag)
	{
		node_scene = new WtNodeFlowScene(&painter, mScene, isSelected, selectedNum);
		node_scene->reorderAllNodes();
		reorderFlag = false;
	}

	node_scene = new WtNodeFlowScene(&painter, mScene, isSelected, selectedNum);

	nodeMap = node_scene->getNodeMap();

	if (isDragging && isSelected)
	{
		auto node = nodeMap[selectedNum];
		moveNode(*node, mTranslateNode);

		auto pointsData = node->flowNodeData().getPointsData();
		for (connectionPointData pointData : pointsData) {
			if (pointData.portShape == PortShape::Bullet)
			{
				painter.drawPolygon(pointData.diamond_out, 4);
			}
			else if (pointData.portShape == PortShape::Diamond)
			{
				painter.drawPolygon(pointData.diamond, 4);
			}
			else if (pointData.portShape == PortShape::Point)
			{
				painter.drawEllipse(pointData.pointRect);
			}
		}
	}

	drawSketchLine(&painter, Wt::WPointF(0, 0), Wt::WPointF(100, 100));
}

bool WtFlowWidget::checkMouseInNodeRect(Wt::WPointF mousePoint, WtFlowNodeData nodeData)
{
	Wt::WPointF bottomRight = Wt::WPointF(nodeData.getNodeBoundingRect().bottomRight().x() + nodeData.getNodeOrigin().x()
		, nodeData.getNodeBoundingRect().bottomRight().y() + nodeData.getNodeOrigin().y());

	Wt::WPointF absTopLeft = Wt::WPointF((nodeData.getNodeOrigin().x() + mTranslate.x()) * mZoomFactor, (nodeData.getNodeOrigin().y() + mTranslate.y()) * mZoomFactor);
	Wt::WPointF absBottomRight = Wt::WPointF((bottomRight.x() + mTranslate.x()) * mZoomFactor, (bottomRight.y() + mTranslate.y()) * mZoomFactor);

	Wt::WRectF absRect = Wt::WRectF(absTopLeft, absBottomRight);

	return absRect.contains(mousePoint);
}

bool WtFlowWidget::checkMouseInHotKey0(Wt::WPointF mousePoint, WtFlowNodeData nodeData)
{
	Wt::WRectF relativeHotkey = nodeData.getHotKey0BoundingRect();
	Wt::WPointF origin = nodeData.getNodeOrigin();

	Wt::WPointF topLeft = Wt::WPointF(relativeHotkey.topLeft().x() + origin.x(), relativeHotkey.topLeft().y() + origin.y());
	Wt::WPointF bottomRight = Wt::WPointF(relativeHotkey.bottomRight().x() + origin.x(), relativeHotkey.bottomRight().y() + origin.y());

	Wt::WRectF absRect = Wt::WRectF(topLeft, bottomRight);

	Wt::WPointF	trueMouse = Wt::WPointF(mousePoint.x() / mZoomFactor - mTranslate.x(), mousePoint.y() / mZoomFactor - mTranslate.y());

	return absRect.contains(trueMouse);
}

bool WtFlowWidget::checkMouseInHotKey1(Wt::WPointF mousePoint, WtFlowNodeData nodeData)
{
	Wt::WRectF relativeHotkey = nodeData.getHotKey1BoundingRect();
	Wt::WPointF origin = nodeData.getNodeOrigin();

	Wt::WPointF topLeft = Wt::WPointF(relativeHotkey.topLeft().x() + origin.x(), relativeHotkey.topLeft().y() + origin.y());
	Wt::WPointF bottomRight = Wt::WPointF(relativeHotkey.bottomRight().x() + origin.x(), relativeHotkey.bottomRight().y() + origin.y());

	Wt::WRectF absRect = Wt::WRectF(topLeft, bottomRight);

	Wt::WPointF	trueMouse = Wt::WPointF(mousePoint.x() / mZoomFactor - mTranslate.x(), mousePoint.y() / mZoomFactor - mTranslate.y());

	return absRect.contains(trueMouse);
}

bool WtFlowWidget::checkMouseInPoints(Wt::WPointF mousePoint, WtFlowNodeData nodeData)
{
	auto pointsData = nodeData.getPointsData();
	Wt::WPointF origin = nodeData.getNodeOrigin();
	Wt::WPointF	trueMouse = Wt::WPointF(mousePoint.x() / mZoomFactor - mTranslate.x(), mousePoint.y() / mZoomFactor - mTranslate.y());

	for (connectionPointData pointData : pointsData)
	{
		if (pointData.portShape == PortShape::Bullet)
		{
			Wt::WPointF topLeft = Wt::WPointF(1, 1);
		}
	}
	return false;
}

void WtFlowWidget::deleteNode(WtNode& n)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr)
	{
		//auto scn = dyno::SceneGraphFactory::instance()->active();
		mScene->deleteNode(nodeData->getNode());
	}
}

void WtFlowWidget::moveNode(WtNode& n, const Wt::WPointF& newLocaton)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr)
	{
		auto node = nodeData->getNode();
		node->setBlockCoord(newLocaton.x(), newLocaton.y());
	}
}

void WtFlowWidget::enableRendering(WtNode& n, bool checked)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr) {
		auto node = nodeData->getNode();
		node->setVisible(checked);
	}
}

void WtFlowWidget::enablePhysics(WtNode& n, bool checked)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());
	if (mEditingEnabled && nodeData != nullptr) {
		auto node = nodeData->getNode();
		node->setActive(checked);
	}
}

void WtFlowWidget::updateForAddNode()
{
	reorderFlag = true;
	update();
	mScene->setFrameNumber(0);
	mScene->reset();
	mMainWindow->updateCanvas();
	mMainWindow->setScene(mScene);
}

void WtFlowWidget::reorderNode()
{
	mZoomFactor = 1;
	reorderFlag = true;
	mTranslate = Wt::WPointF(0, 0);
	update();
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