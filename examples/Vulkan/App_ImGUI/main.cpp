#define _ENABLE_EXTENDED_ALIGNED_STORAGE
#include "ImGUI/VkApp.h"

#include "SceneGraph.h"

#define ENABLE_VALIDATION true

using namespace dyno;
std::shared_ptr<dyno::SceneGraph> createScene()
{
	auto scene = std::make_shared<dyno::SceneGraph>();

	return scene;
}

// OS specific macros for the example main entry points
#if defined(_WIN32)
// Windows entry point
VkApp *vulkanExample;
LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (vulkanExample != NULL)
	{
		vulkanExample->handleMessages(hWnd, uMsg, wParam, lParam);
	}
	return (DefWindowProc(hWnd, uMsg, wParam, lParam));
}
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int)
{
	VkSystem::instance()->initialize();
	vulkanExample = new VkApp(ENABLE_VALIDATION);
	vulkanExample->setSceneGraph(createScene());
	vulkanExample->setWindowTitle("Rigid body simulation");
	vulkanExample->initVulkan();
	vulkanExample->setupWindow(hInstance, WndProc);
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
// Android entry point
VkApp *vulkanExample;
void android_main(android_app* state)
{
	px::VkSystem::instance()->initialize();
	vulkanExample = new VkApp(ENABLE_VALIDATION);
	vulkanExample->setWindowTitle("Rigid body simulation");
	vulkanExample->setSceneGraph(createScene());
	state->userData = vulkanExample;
	state->onAppCmd = VkApp::handleAppCommand;
	state->onInputEvent = VkApp::handleAppInput;
	androidApp = state;
	vks::android::getDeviceConfig();
	vulkanExample->renderLoop();
	delete(vulkanExample);
}
#endif

