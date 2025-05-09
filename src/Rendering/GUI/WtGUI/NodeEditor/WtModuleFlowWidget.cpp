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
	if (mNode == nullptr)
		return;

	isDragging = true;
	mLastMousePos = Wt::WPointF(event.widget().x, event.widget().y);
	mLastDelta = Wt::WPointF(0, 0);
	if (!checkMouseInAllRect(Wt::WPointF(event.widget().x, event.widget().y)))
	{
		selectType = -1;
		selectedNum = 0;
		canMoveNode = false;
		update();
	}
	if (selectType > 0)
	{
		auto origin = moduleMap[selectedNum]->flowNodeData().getNodeOrigin();
		mTranslateNode = Wt::WPointF(origin.x(), origin.y());
		if (checkMouseInPoints(mLastMousePos, moduleMap[selectedNum]->flowNodeData(), PortState::out))
		{
			drawLineFlag = true;

			// change module
			if (moduleMap.find(selectedNum) != moduleMap.end())
			{
				mOutModule = moduleMap.find(selectedNum)->second->getModule();
			}


		/*	if (outPoint.portType == PortType::In)
			{
				auto existConnection = moduleMap[selectedNum]->getIndexConnection(outPoint.portIndex);
				if (existConnection != nullptr)
				{
					auto outNode = existConnection->getNode(PortType::Out);
					for (auto it = mScene->begin(); it != mScene->end(); it++)
					{
						auto m = it.get();
						auto node = moduleMap[m->objectId()];
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
			}*/
		}
		else
		{
			selectType = 2;
		}
	}
}

void WtModuleFlowWidget::onMouseMove(const Wt::WMouseEvent& event)
{
	if (mNode == nullptr)
		return;

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
		auto mlist = mNode->getModuleList();
		auto mousePoint = Wt::WPointF(event.widget().x, event.widget().y);
		for (const auto& module : mlist)
		{
			if (moduleMap.find(module->objectId()) != moduleMap.end())
			{
				auto m = moduleMap[module->objectId()];
				auto moduleData = m->flowNodeData();
				if (checkMouseInRect(mousePoint, moduleData) && selectType != 2)
				{
					selectType = 1;
					//connectionOutNode = node;
					selectedNum = module->objectId();
					canMoveNode = true;
					update();
					break;
				}
			}
			
		}
	}
}

void WtModuleFlowWidget::onMouseWentUp(const Wt::WMouseEvent& event)
{
	if (mNode == nullptr)
		return;

	isDragging = false;
	mLastDelta = Wt::WPointF(0, 0);
	Wt::WPointF mouseWentUpPosition = Wt::WPointF(event.widget().x, event.widget().y);
	if (selectType > 0)
	{
		auto node = moduleMap[selectedNum];
		auto nodeData = node->flowNodeData();

		_selectNodeSignal.emit(selectedNum);

		Wt::WPointF mousePoint = Wt::WPointF(event.widget().x, event.widget().y);
		if (!checkMouseInAllRect(mousePoint) && selectType == 2)
		{
			selectType = -1;
			selectedNum = 0;
			canMoveNode = false;
			update();
			_updateCanvas.emit();
		}
	}
	else
	{
		_selectNodeSignal.emit(-1);
	}

	drawLineFlag = false;
	update();
}

void WtModuleFlowWidget::onKeyWentDown()
{
	if (mNode == nullptr)
		return;

	return;
}

void WtModuleFlowWidget::setNode(std::shared_ptr<dyno::Node> node)
{
	mNode = node;
	update();
}

void WtModuleFlowWidget::deleteModule()
{
	
}

void WtModuleFlowWidget::moveModule(WtNode& n, const Wt::WPointF& newLocation)
{
	WtModuleWidget* mw = dynamic_cast<WtModuleWidget*>(n.nodeDataModel());

	auto m = mw == nullptr ? nullptr : mw->getModule();

	if (m != nullptr)
	{
		m->setBlockCoord(newLocation.x(), newLocation.y());
	}
}

void WtModuleFlowWidget::paintEvent(Wt::WPaintDevice* paintDevice)
{
	Wt::WPainter painter(paintDevice);
	painter.scale(mZoomFactor, mZoomFactor);
	painter.translate(mTranslate);

	if (mNode != nullptr)
	{
		if (reorderFlag)
		{
			mModuleFlowScene = new WtModuleFlowScene(&painter, mNode);
			mModuleFlowScene->reorderAllModules();
			reorderFlag = false;
		}

		mModuleFlowScene = new WtModuleFlowScene(&painter, mNode);
		moduleMap = mModuleFlowScene->getNodeMap();
	}

	if (isDragging && selectType > 0 && !drawLineFlag)
	{
		auto node = moduleMap[selectedNum];
		moveModule(*node, mTranslateNode);
	}

	if (drawLineFlag)
	{
		//drawSketchLine(&painter, sourcePoint, sinkPoint);
	}
}

bool WtModuleFlowWidget::checkMouseInAllRect(Wt::WPointF mousePoint)
{
	auto mlist = mNode->getModuleList();
	for (const auto& module : mlist)
	{
		if (moduleMap.find(module->objectId()) != moduleMap.end())
		{
			auto m = moduleMap[module->objectId()];
			auto moduleData = m->flowNodeData();
			if (checkMouseInRect(mousePoint, moduleData))
			{
				selectedNum = module->objectId();
				canMoveNode = true;
				return true;
			}
		}
		return false;
	}
}
