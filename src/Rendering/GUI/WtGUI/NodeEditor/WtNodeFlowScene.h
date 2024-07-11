#pragma once

#include "WtFlowScene.h"

#include "SceneGraph.h"
#include "FBase.h"

using dyno::SceneGraph;

class WtNodeFlowScene : public WtFlowScene
{
	WtNodeFlowScene(WObject* parent = nullptr);
	~WtNodeFlowScene();
};