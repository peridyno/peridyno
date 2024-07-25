#include "WtFlowScene.h"

WtFlowScene::WtFlowScene() {}

WtFlowScene::WtFlowScene(std::shared_ptr<WtDataModelRegistry> registry, Wt::WPainter* painter)
	: _registry(std::move(registry))
{
}

WtFlowScene::~WtFlowScene()
{
}