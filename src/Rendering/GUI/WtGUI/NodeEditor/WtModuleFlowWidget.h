#pragma once

#include "WtFlowWidget.h"

class WtModuleFlowWidget : WtFlowWidget
{
public:
	WtModuleFlowWidget(std::shared_ptr<dyno::SceneGraph> scene);
	~WtModuleFlowWidget();
};