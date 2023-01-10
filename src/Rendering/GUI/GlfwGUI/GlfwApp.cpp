#include "GlfwApp.h"

#include "GlfwRenderWindow.h"

namespace dyno 
{
	GlfwApp::GlfwApp(int argc /*= 0*/, char **argv /*= NULL*/)
	{
	}

	GlfwApp::~GlfwApp()
	{
	}

	void GlfwApp::createWindow(int width, int height, bool usePlugin)
	{
		std::cout << "createWindow() will be depreciated in the near future, please use resize() instead" << std::endl;

		mRenderWindow = std::make_shared<GlfwRenderWindow>();

		mRenderWindow->createWindow(width, height);
	}

	void GlfwApp::createWindow(int width, int height, std::shared_ptr<RenderEngine> engine)
	{
		mRenderWindow = std::make_shared<GlfwRenderWindow>();
		mRenderWindow->setRenderEngine(engine);
		mRenderWindow->createWindow(width, height);
	}

	void GlfwApp::resize(int width, int height)
	{
		mRenderWindow = std::make_shared<GlfwRenderWindow>();

		mRenderWindow->createWindow(width, height);
	}

	void GlfwApp::mainLoop()
	{
		mRenderWindow->mainLoop();
	}
}