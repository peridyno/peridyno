#include "UbiApp.h"

namespace dyno
{
	UbiApp::UbiApp(GUIType type)
	{
		if (type == GUIType::GUI_GLFW)
		{
			mApp = new GlfwApp;
		}

		if (type == GUIType::GUI_QT)
		{
#if (defined QT_GUI_SUPPORTED)
			mApp = new QtApp;
#else
			mApp = new GlfwApp;
#endif
		}

		if (type == GUIType::GUI_WT)
		{
#if (defined WT_GUI_SUPPORTED)
			mApp = new WtApp;
#else
			mApp = new GlfwApp;
#endif
		}
	}

	UbiApp::~UbiApp()
	{
		delete mApp;
	}

	void UbiApp::initialize(int width, int height, bool usePlugin /*= false*/)
	{
		mApp->initialize(width, height, usePlugin);
	}

	void UbiApp::mainLoop()
	{
		mApp->mainLoop();
	}

}
