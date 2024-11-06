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
}

void WtFlowWidget::onMouseMove(const Wt::WMouseEvent& event)
{
	if (isDragging)
	{
		Wt::WPointF delta = Wt::WPointF(event.dragDelta().x, event.dragDelta().y);
		mTranlate = Wt::WPointF(mTranlate.x() + delta.x() - mLastDelta.x(), mTranlate.y() + delta.y() - mLastDelta.y());
		update();
		mLastDelta = delta;
	}
}

void WtFlowWidget::onMouseWentUp(const Wt::WMouseEvent& event)
{
	isDragging = false;
	mLastDelta = Wt::WPointF(0, 0);
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
	//painter.translate(mTranlate);

	node_scene = new WtNodeFlowScene(&painter, mScene);

	nodeMap = node_scene->getNodeMap();

	for (auto it = mScene->begin(); it != mScene->end(); it++)
	{
		auto m = it.get();
		auto node = nodeMap[m->objectId()];
		if (m->objectId() == 2 || m->objectId() == 364)
		{
			moveNode(*node, mTranlate);
			auto nodeData = node->flowNodeData();
			auto p = nodeData.getNodeBoundingRect().bottomRight();
			auto p1 = nodeData.getNodeOrigin();
			std::cout << p1.x() << std::endl;
			std::cout << p1.y() << std::endl;
			std::cout << p.x() << std::endl;
			std::cout << p.y() << std::endl;

		}
	}
}

void WtFlowWidget::moveNode(WtNode& n, const Wt::WPointF& newLocaton)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (true && nodeData != nullptr)
	{
		auto node = nodeData->getNode();
		node->setBlockCoord(newLocaton.x(), newLocaton.y());
	}
}