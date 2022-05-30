#include <QMap>
#include <QDebug>
#include "QtApp.h"
#include "PMainWindow.h"
#include "Log.h"
//#include "Rendering/OpenGLContext.h"
#include "SceneGraphFactory.h"

#include <GLRenderEngine.h>

namespace dyno {
    QtApp::QtApp(int argc, char **argv)
    {
        m_mainWindow = nullptr;
        m_app = std::make_shared<QApplication>(argc, argv);

		//To resolver the error "Cannot queue arguments of type of Log::Message" for multi-thread applications
		qRegisterMetaType<Log::Message>("Log::Message");
    }

    QtApp::~QtApp()
    {

    }

    void QtApp::createWindow(int width, int height)
    {
        m_mainWindow = std::make_shared<PMainWindow>(renderEngine().get());
        m_mainWindow->resize(width, height);
    }

    void QtApp::mainLoop()
    {
        m_mainWindow->show();
        m_app->exec();
    }

	void QtApp::setRenderEngine(std::shared_ptr<RenderEngine> engine)
	{
        //TODO: replace the default render engine with an new one in runtime.
        mRenderEngine = engine;
	}

	void QtApp::setSceneGraph(std::shared_ptr<SceneGraph> scn)
	{
        SceneGraphFactory::instance()->pushScene(scn);
	}

	std::shared_ptr<RenderEngine> QtApp::renderEngine()
	{
		if (mRenderEngine == nullptr)
			mRenderEngine = std::make_shared<GLRenderEngine>();

		return mRenderEngine;
	}

}
