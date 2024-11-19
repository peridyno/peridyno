#include "WtFlowWidget.h"

WtFlowWidget::WtFlowWidget(std::shared_ptr<dyno::SceneGraph> scene) :Wt::WPaintedWidget()
{
	mZoomFactor = 1.0;
	mScene = scene;
	resize(600, 900);

	setPreferredMethod(Wt::RenderMethod::HtmlCanvas);

	this->mouseWentDown().connect(this, &WtFlowWidget::onMouseWentDown);
	this->mouseMoved().connect(this, &WtFlowWidget::onMouseMove);
	this->mouseWentUp().connect(this, &WtFlowWidget::onMouseWentUp);
	this->mouseWheel().connect(this, &WtFlowWidget::onMouseWheel);
	this->keyWentDown().connect(this, &WtFlowWidget::onKeyWentDown);
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
			std::cout << "!!" << std::endl;
			//enableRendering(*node, false);

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

void WtFlowWidget::onKeyWentDown(const Wt::WKeyEvent& event)
{
	std::cout << "aaa" << std::endl;
	if (event.key() == Wt::Key::Up)
	{
		std::cout << "!!!" << std::endl;
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

	node_scene = new WtNodeFlowScene(&painter, mScene, isSelected, selectedNum);

	if (reorderFlag)
	{
		node_scene->reorderAllNodes();
		reorderFlag = false;
		update();
	}

	nodeMap = node_scene->getNodeMap();

	if (isDragging && isSelected)
	{
		auto node = nodeMap[selectedNum];
		moveNode(*node, mTranslateNode);

		enableRendering(*node, false);

	}

	for (auto it = mScene->begin(); it != mScene->end(); it++)
	{
		auto node = it.get();
		std::cout << node->isVisible() << std::endl;
	}
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
	Wt::WPointF bottomRight = Wt::WPointF(nodeData.getHotKey0BoundingRect().bottomRight().x() + nodeData.getNodeOrigin().x()
		, nodeData.getHotKey0BoundingRect().bottomRight().y() + nodeData.getNodeOrigin().y());


	Wt::WPointF absTopLeft = Wt::WPointF((nodeData.getNodeOrigin().x() + mTranslate.x()) * mZoomFactor, (nodeData.getNodeOrigin().y() + mTranslate.y()) * mZoomFactor);
	Wt::WPointF absBottomRight = Wt::WPointF((bottomRight.x() + mTranslate.x()) * mZoomFactor, (bottomRight.y() + mTranslate.y()) * mZoomFactor);

	Wt::WRectF absRect = Wt::WRectF(absTopLeft, absBottomRight);

	return absRect.contains(mousePoint);

}

bool WtFlowWidget::checkMouseInHotKey1(Wt::WPointF mousePoint, WtFlowNodeData nodeData)
{
	return true;
}


void WtFlowWidget::addNode(WtNode& n)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr)
	{
		auto scn = dyno::SceneGraphFactory::instance()->active();
		scn->addNode(nodeData->getNode());
	}
}

void WtFlowWidget::deleteNode(WtNode& n)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr)
	{
		auto scn = dyno::SceneGraphFactory::instance()->active();
		scn->deleteNode(nodeData->getNode());
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