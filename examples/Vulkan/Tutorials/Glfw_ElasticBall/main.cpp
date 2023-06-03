#include <GlfwApp.h>

#include <SceneGraph.h>
#include <GLSurfaceVisualModule.h>

#include "ElasticBody/ElasticBody.h"

using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	auto scene = std::make_shared<SceneGraph>();

	auto ball = scene->addNode(std::make_shared<ElasticBody>());

	ball->loadFromFile(getAssetPath() + "models/standard_sphere.1");

	auto clothRender = std::make_shared<GLSurfaceVisualModule>();
	ball->stateTopology()->connect(clothRender->inTriangleSet());
	ball->graphicsPipeline()->pushModule(clothRender);

	return scene;
}

int main(int, char**)
{
	VkSystem::instance()->initialize();

	GlfwApp window;
	window.initialize(1024, 768);
	window.setSceneGraph(createScene());
	window.mainLoop();
	return 0;
}
