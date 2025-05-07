#include "WtModuleFlowWidget.h"


WtModuleFlowWidget::WtModuleFlowWidget(std::shared_ptr<dyno::SceneGraph> scene)
	: WtFlowWidget(scene)
{
	this->mouseWentDown().connect(this, &WtModuleFlowWidget::onMouseWentDown);
	this->mouseMoved().connect(this, &WtModuleFlowWidget::onMouseMove);
	this->mouseWentUp().connect(this, &WtModuleFlowWidget::onMouseWentUp);
}

WtModuleFlowWidget::~WtModuleFlowWidget(){}

void WtModuleFlowWidget::onMouseWentDown(const Wt::WMouseEvent& event)
{
	isDragging = true;
	mLastMousePos = Wt::WPointF(event.widget().x, event.widget().y);
	mLastDelta = Wt::WPointF(0, 0);
}

void WtModuleFlowWidget::onMouseMove(const Wt::WMouseEvent& event)
{
	if (isDragging)
	{
		Wt::WPointF delta = Wt::WPointF(event.dragDelta().x, event.dragDelta().y);
		mTranslate = Wt::WPointF(mTranslate.x() + delta.x() - mLastDelta.x(), mTranslate.y() + delta.y() - mLastDelta.y());
		update();
		mLastDelta = delta;
	}
}

void WtModuleFlowWidget::onMouseWentUp(const Wt::WMouseEvent& event)
{
	isDragging = false;
	mLastDelta = Wt::WPointF(0, 0);
	Wt::WPointF mouseWentUpPosition = Wt::WPointF(event.widget().x, event.widget().y);

	drawLineFlag = false;
	update();
}

void WtModuleFlowWidget::onKeyWentDown()
{
	return;
}

void WtModuleFlowWidget::setNode(std::shared_ptr<dyno::Node> node)
{
	mNode = node;
	update();
}

void WtModuleFlowWidget::paintEvent(Wt::WPaintDevice* paintDevice)
{
	Wt::WPainter painter(paintDevice);
	painter.scale(mZoomFactor, mZoomFactor);
	painter.translate(mTranslate);

	/*painter.setBrush(Wt::WBrush(Wt::WColor(Wt::StandardColor::Blue)));
	painter.drawRect(0, 0, 100, 50);*/
	
	//if (reorderFlag)
	//{
	//	mModuleFlowScene = new WtModuleFlowScene(&painter, mNode);
	//	//mModuleFlowScene->reorderAllNodes();
	//	reorderFlag = false;
	//}

	mModuleFlowScene = new WtModuleFlowScene(&painter, mNode);
}
