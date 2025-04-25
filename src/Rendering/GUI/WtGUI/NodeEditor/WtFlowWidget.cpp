#include "WtFlowWidget.h"
#include <Wt/WEnvironment.h>
#include <Wt/WApplication.h>
#include <Wt/WMessageBox.h>

WtFlowWidget::WtFlowWidget(std::shared_ptr<dyno::SceneGraph> scene) :Wt::WPaintedWidget()
{
	mZoomFactor = 1.0;
	mScene = scene;

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
	if (selectType > 0)
	{
		auto origin = nodeMap[selectedNum]->flowNodeData().getNodeOrigin();
		mTranslateNode = Wt::WPointF(origin.x(), origin.y());
		if (checkMouseInPoints(mLastMousePos, nodeMap[selectedNum]->flowNodeData(), PortState::out))
		{
			drawLineFlag = true;
			for (auto it = mScene->begin(); it != mScene->end(); it++)
			{
				auto m = it.get();
				if (m->objectId() == selectedNum)
				{
					mOutNode = m;
				}
			}
			if (outPoint.portType == PortType::In)
			{
				auto existConnection = nodeMap[selectedNum]->getIndexConnection(outPoint.portIndex);
				if (existConnection != nullptr)
				{
					auto outNode = existConnection->getNode(PortType::Out);
					for (auto it = mScene->begin(); it != mScene->end(); it++)
					{
						auto m = it.get();
						auto node = nodeMap[m->objectId()];
						auto outPortIndex = existConnection->getPortIndex(PortType::Out);
						auto exportPortsData = outNode->flowNodeData().getPointsData();
						connectionPointData exportPointData;
						for (auto point : exportPortsData)
						{
							if (point.portIndex == outPortIndex)
							{
								exportPointData = point;
								break;
							}
						}

						if (node == outNode)
						{
							for (auto it = sceneConnections.begin(); it != sceneConnections.end(); )
							{
								if (it->exportNode == mOutNode && it->inportNode == m && it->inPoint.portIndex == outPoint.portIndex && it->outPoint.portIndex == exportPointData.portIndex)
								{
									it = sceneConnections.erase(it);
								}
								else
								{
									++it;
								}
							}

							disconnect(m, mOutNode, outPoint, exportPointData, nodeMap[selectedNum], outNode);
							sourcePoint = getPortPosition(outNode->flowNodeData().getNodeOrigin(), exportPointData);

						}
					}

				}
			}
		}
		else
		{
			// selectType = 2: selected & drag
			// selectType = 1: mouse move node
			// selectType = -1: no select
			selectType = 2;
		}
	}

}

void WtFlowWidget::onMouseMove(const Wt::WMouseEvent& event)
{
	sinkPoint = Wt::WPointF(event.widget().x / mZoomFactor - mTranslate.x(), event.widget().y / mZoomFactor - mTranslate.y());
	if (isDragging && selectType < 0)
	{
		Wt::WPointF delta = Wt::WPointF(event.dragDelta().x, event.dragDelta().y);
		mTranslate = Wt::WPointF(mTranslate.x() + delta.x() - mLastDelta.x(), mTranslate.y() + delta.y() - mLastDelta.y());
		update();
		mLastDelta = delta;
	}
	else if (isDragging && selectType > 0)
	{
		Wt::WPointF delta = Wt::WPointF(event.dragDelta().x, event.dragDelta().y);
		mTranslateNode = Wt::WPointF(mTranslateNode.x() + delta.x() - mLastDelta.x(), mTranslateNode.y() + delta.y() - mLastDelta.y());
		update();
		mLastDelta = delta;
	}
	else
	{
		auto mousePoint = Wt::WPointF(event.widget().x, event.widget().y);
		if (checkMouseInAllNodeRect(mousePoint) && selectType != 2)
		{
			selectType = 1;
			update();
		}
		else
		{
			if (selectType != 2)
			{
				selectType = -1;
				selectedNum = 0;
				canMoveNode = false;
				update();
			}
		}
	}
}

void WtFlowWidget::onMouseWentUp(const Wt::WMouseEvent& event)
{
	isDragging = false;
	mLastDelta = Wt::WPointF(0, 0);
	mTranslateNode = Wt::WPointF(0, 0);
	Wt::WPointF mouseWentUpPosition = Wt::WPointF(event.widget().x, event.widget().y);
	if (selectType > 0)
	{
		auto node = nodeMap[selectedNum];
		auto nodeData = node->flowNodeData();

		_selectNodeSignal.emit(selectedNum);

		Wt::WPointF mousePoint = Wt::WPointF(event.widget().x, event.widget().y);
		if (!checkMouseInAllNodeRect(mousePoint) && selectType == 2)
		{
			selectType = -1;
			selectedNum = 0;
			canMoveNode = false;
			update();
			_updateCanvas.emit();
		}

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
			//mMainWindow->updateCanvas();
			_updateCanvas.emit();
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
			//mMainWindow->updateCanvas();
			_updateCanvas.emit();
			update();
		}
	}
	else
	{
		_selectNodeSignal.emit(-1);
	}

	if (drawLineFlag = true)
	{
		for (auto it = mScene->begin(); it != mScene->end(); it++)
		{
			auto m = it.get();
			auto node = nodeMap[m->objectId()];
			auto nodeData = node->flowNodeData();
			if (checkMouseInPoints(mouseWentUpPosition, nodeData, PortState::in))
			{
				std::cout << "node data" << std::endl;
				auto connectionInNode = node;

				if (outPoint.portType == PortType::Out)
				{
					WtConnection connection(outPoint.portType, *connectionOutNode, outPoint.portIndex);
					WtInteraction interaction(*connectionInNode, connection, *node_scene, inPoint, outPoint, m, mOutNode);
					if (interaction.tryConnect())
					{
						sceneConnection temp;
						temp.exportNode = m;
						temp.inportNode = mOutNode;
						temp.inPoint = inPoint;
						temp.outPoint = outPoint;
						sceneConnections.push_back(temp);
						update();
					}
				}

			}
		}
	}

	drawLineFlag = false;
	update();
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
	if (selectType > 0)
	{
		auto node = nodeMap[selectedNum];
		deleteNode(*node);
		selectType = -1;
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

void WtFlowWidget::setSelectNode(std::shared_ptr<dyno::Node> node)
{
	selectType = 2;
	selectedNum = node->objectId();
	update();
}

void WtFlowWidget::paintEvent(Wt::WPaintDevice* paintDevice)
{
	Wt::WPainter painter(paintDevice);
	painter.scale(mZoomFactor, mZoomFactor);
	painter.translate(mTranslate);

	if (reorderFlag)
	{
		node_scene = new WtNodeFlowScene(&painter, mScene, selectType, selectedNum);
		node_scene->reorderAllNodes();
		reorderFlag = false;
	}

	node_scene = new WtNodeFlowScene(&painter, mScene, selectType, selectedNum);

	nodeMap = node_scene->getNodeMap();

	if (isDragging && selectType > 0 && !drawLineFlag)
	{
		auto node = nodeMap[selectedNum];
		moveNode(*node, mTranslateNode);
	}

	if (drawLineFlag)
	{
		drawSketchLine(&painter, sourcePoint, sinkPoint);
	}
}

bool WtFlowWidget::checkMouseInAllNodeRect(Wt::WPointF mousePoint)
{
	for (auto it = mScene->begin(); it != mScene->end(); it++)
	{
		auto m = it.get();
		auto node = nodeMap[m->objectId()];
		auto nodeData = node->flowNodeData();
		if (checkMouseInNodeRect(mousePoint, nodeData))
		{
			connectionOutNode = node;
			selectedNum = m->objectId();
			canMoveNode = true;
			return true;
		}
	}
	return false;
}

bool WtFlowWidget::checkMouseInNodeRect(Wt::WPointF mousePoint, WtFlowNodeData nodeData)
{
	Wt::WPointF bottomRight = Wt::WPointF(nodeData.getNodeBoundingRect().bottomRight().x() + nodeData.getNodeOrigin().x()
		, nodeData.getNodeBoundingRect().bottomRight().y() + nodeData.getNodeOrigin().y());

	Wt::WPointF absTopLeft = Wt::WPointF((nodeData.getNodeOrigin().x() + mTranslate.x() - 10) * mZoomFactor, (nodeData.getNodeOrigin().y() + mTranslate.y() - 10) * mZoomFactor);
	Wt::WPointF absBottomRight = Wt::WPointF((bottomRight.x() + mTranslate.x() + 10) * mZoomFactor, (bottomRight.y() + mTranslate.y() + 10) * mZoomFactor);

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

void WtFlowWidget::deleteNode(WtNode& n)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr)
	{
		auto node = nodeData->getNode();

		for (auto c : sceneConnections)
		{
			if (c.exportNode == node || c.inportNode == node)
			{
				Wt::WMessageBox::show("Error",
					"<p>Please disconnect before deleting the node </p>",
					Wt::StandardButton::Ok);
				return;
			}
		}

		mScene->deleteNode(node);
	}
}

void WtFlowWidget::disconnectionsFromNode(WtNode& node)
{

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
	//reorderFlag = true;
	update();
	mScene->setFrameNumber(0);
	mScene->reset();
	_updateCanvas.emit();
	//mMainWindow->updateCanvas();
	//mMainWindow->setScene(mScene);
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



void WtFlowWidget::disconnect(std::shared_ptr<Node> exportNode, std::shared_ptr<Node> inportNode, connectionPointData inPoint, connectionPointData outPoint, WtNode* inWtNode, WtNode* outWtNode)
{
	auto inportIndex = inPoint.portIndex;
	if (inPoint.portShape == PortShape::Diamond || inPoint.portShape == PortShape::Bullet)
	{
		exportNode->disconnect(inportNode->getImportNodes()[inportIndex]);
	}
	else if (inPoint.portShape == PortShape::Point)
	{
		auto outFieldNum = 0;
		auto outPoints = outWtNode->flowNodeData().getPointsData();
		for (auto point : outPoints)
		{
			if (point.portShape == PortShape::Point)
			{
				outFieldNum = point.portIndex;
				break;
			}
		}

		auto field = exportNode->getOutputFields()[outPoint.portIndex - outFieldNum];

		if (field != NULL)
		{
			auto node_data = inWtNode->flowNodeData();

			auto points = node_data.getPointsData();

			int fieldNum = 0;

			for (auto point : points)
			{
				if (point.portType == PortType::In)
				{
					if (point.portShape == PortShape::Bullet || point.portShape == PortShape::Diamond)
					{
						fieldNum++;
					}
				}
			}

			auto inField = inportNode->getInputFields()[inPoint.portIndex - fieldNum];

			field->disconnect(inField);
		}
	}
}