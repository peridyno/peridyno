#include "WtModuleFlowWidget.h"
#include "WtInteraction.h"
#include <Wt/WMessageBox.h>

WtModuleFlowWidget::WtModuleFlowWidget(std::shared_ptr<dyno::SceneGraph> scene)
	: WtFlowWidget(scene)
{
	this->mouseWentDown().connect(this, &WtModuleFlowWidget::onMouseWentDown);
	this->mouseMoved().connect(this, &WtModuleFlowWidget::onMouseMove);
	this->mouseWentUp().connect(this, &WtModuleFlowWidget::onMouseWentUp);
}

WtModuleFlowWidget::~WtModuleFlowWidget() {}

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

			if (outPoint.portType == PortType::In)
			{
				auto existConnection = moduleMap[selectedNum]->getIndexConnection(outPoint.portIndex);

				if (existConnection != nullptr)
				{
					auto outNode = existConnection->getNode(PortType::Out);

					for (auto m : moduleMap)
					{
						auto node = m.second;
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
							for (auto it = nodeConnections.begin(); it != nodeConnections.end(); )
							{
								if (it->exportModule == mOutModule && it->inportModule == m.second->getModule() && it->inPoint.portIndex == outPoint.portIndex && it->outPoint.portIndex == exportPointData.portIndex)
								{
									it = nodeConnections.erase(it);
								}
								else
								{
									++it;
								}
							}

							disconnect(m.second->getModule(), mOutModule, outPoint, exportPointData, moduleMap[selectedNum], outNode);
							sourcePoint = getPortPosition(outNode->flowNodeData().getNodeOrigin(), exportPointData);
						}
					}
				}
			}
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
					connectionOutNode = m;
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

		_selectModuleSignal.emit(node->getModule());

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
		_selectModuleSignal.emit(nullptr);
	}

	if (drawLineFlag = true)
	{
		for (auto module : moduleMap)
		{
			auto node = module.second;
			auto nodeData = node->flowNodeData();
			if (checkMouseInPoints(mouseWentUpPosition, nodeData, PortState::in))
			{
				auto connectionInNode = node;

				if (outPoint.portType == PortType::Out)
				{
					WtConnection connection(outPoint.portType, *connectionOutNode, outPoint.portIndex);
					WtInteraction interaction(*connectionInNode, connection, inPoint, outPoint, node->getModule(), mOutModule);
					if (interaction.tryConnect())
					{
						sceneConnection temp;
						temp.exportModule = node->getModule();
						temp.inportModule = mOutModule;
						temp.inPoint = inPoint;
						temp.outPoint = outPoint;
						nodeConnections.push_back(temp);
						update();
					}
				}
			}
		}
	}

	drawLineFlag = false;
	update();
}

void WtModuleFlowWidget::onKeyWentDown()
{
	if (mNode == nullptr)
		return;

	if (selectType > 0)
	{
		auto module = moduleMap[selectedNum]->getModule();
		deleteModule(module);
		selectType = -1;
		selectedNum = 0;
		updateAll();
	}
}

void WtModuleFlowWidget::setNode(std::shared_ptr<dyno::Node> node)
{
	mNode = node;
	update();
}

void WtModuleFlowWidget::addModule(std::shared_ptr<dyno::Module> new_module)
{
	if (mModuleFlowScene != nullptr)
		mModuleFlowScene->addModule(new_module);

	updateAll();
}

void WtModuleFlowWidget::deleteModule(std::shared_ptr<dyno::Module> delete_module)
{
	if (mModuleFlowScene != nullptr)
	{
		for (auto c : nodeConnections)
		{
			std::cout << c.exportModule->getName() << std::endl;

			if (c.exportModule == delete_module || c.inportModule == delete_module)
			{
				Wt::WMessageBox::show("Error",
					"<p>Please disconnect before deleting the module </p>",
					Wt::StandardButton::Ok);
				return;
			}
		}

		mModuleFlowScene->deleteModule(delete_module);
	}

	updateAll();
}


void WtModuleFlowWidget::moveModule(WtNode& n, const Wt::WPointF& newLocation)
{
	auto m = n.getModule();

	if (m != nullptr)
	{
		m->setBlockCoord(newLocation.x(), newLocation.y());
	}
}

void WtModuleFlowWidget::showResetPipeline()
{
	pipelineType = PipelineType::Reset;
	update();
	reorderFlag = true;
}

void WtModuleFlowWidget::showAnimationPipeline()
{
	pipelineType = PipelineType::Animation;
	update();
	reorderFlag = true;
}

void WtModuleFlowWidget::showGraphicsPipeline()
{
	pipelineType = PipelineType::Graphics;
	update();
	reorderFlag = true;
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
			mModuleFlowScene = new WtModuleFlowScene(&painter, mNode, pipelineType);
			mModuleFlowScene->reorderAllModules();
			reorderFlag = false;
		}

		mModuleFlowScene = new WtModuleFlowScene(&painter, mNode, pipelineType);
		moduleMap = mModuleFlowScene->getNodeMap();
	}

	if (isDragging && selectType > 0 && !drawLineFlag)
	{
		auto node = moduleMap[selectedNum];
		moveModule(*node, mTranslateNode);
	}

	if (drawLineFlag)
	{
		drawSketchLine(&painter, sourcePoint, sinkPoint);
	}
}

bool WtModuleFlowWidget::checkMouseInAllRect(Wt::WPointF mousePoint)
{
	for (auto module : moduleMap)
	{
		auto moduleData = module.second->flowNodeData();

		if (checkMouseInRect(mousePoint, moduleData))
		{
			selectedNum = module.second->getModule()->objectId();
			canMoveNode = true;
			return true;
		}
	}
	return false;
}

void WtModuleFlowWidget::disconnect(std::shared_ptr<Module> exportModule, std::shared_ptr<Module> inportModule, connectionPointData inPoint, connectionPointData outPoint, WtNode* inWtNode, WtNode* outWtNode)
{
	auto inportIndex = inPoint.portIndex;

	if (inPoint.portShape == PortShape::Point)
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

		auto field = exportModule->getOutputFields()[outPoint.portIndex - outFieldNum];

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

			auto inField = inportModule->getInputFields()[inPoint.portIndex - fieldNum];

			field->disconnect(inField);
		}
	}
}