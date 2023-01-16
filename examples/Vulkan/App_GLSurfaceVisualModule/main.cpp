#include <GlfwApp.h>

#include <SceneGraph.h>
#include <GLSurfaceVisualModule.h>

#include "Cloth.h"

using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	auto scene = std::make_shared<SceneGraph>();

	auto cloth = scene->addNode(std::make_shared<Cloth>());

	cloth->loadObjFile("../../../data/bunny/sparse_bunny_mesh.obj");

	auto clothRender = std::make_shared<dyno::GLSurfaceVisualModule>();
	cloth->stateTopology()->connect(clothRender->inTriangleSet());
	cloth->graphicsPipeline()->pushModule(clothRender);

	return scene;
}

int main(int, char**)
{
	VkSystem* vkSys = VkSystem::instance();
	vkSys->enabledInstanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
	vkSys->enabledInstanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

	vkSys->enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
	vkSys->enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef WIN32
	vkSys->enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
	vkSys->enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
	vkSys->enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
	vkSys->enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif

	vkSys->initialize(true);

	GlfwApp app;

	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	
	app.mainLoop();
	return 0;
}
