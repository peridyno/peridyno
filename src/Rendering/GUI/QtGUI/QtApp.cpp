#include <QMap>
#include <QDebug>
#include <QFile>
#include "QtApp.h"
#include "PMainWindow.h"
#include "POpenGLWidget.h"
#include "Log.h"

#include "SceneGraphFactory.h"
#include "Plugin/PluginManager.h"

#include <OrbitCamera.h>

namespace dyno {
    QtApp::QtApp(int argc, char **argv)
    {
#ifdef CUDA_BACKEND
        auto status = cudaSetDevice(0);
		if (status != cudaSuccess) {
			fprintf(stderr, "CUDA initialization failed!  Do you have a CUDA-capable GPU installed?");
			exit(0);
		}
        cudaFree(0);
#endif // CUDA_BACKEND

        mMainWindow = nullptr;
        mQApp = std::make_shared<QApplication>(argc, argv);

		//To resolver the error "Cannot queue arguments of type of Log::Message" for multi-thread applications
		qRegisterMetaType<Log::Message>("Log::Message");
    }

    QtApp::~QtApp()
    {

    }

	void QtApp::initialize(int width, int height, bool usePlugin)
    {
        //A hack to address the slow launching problem

		if (usePlugin)
		{
#ifdef NDEBUG
			PluginManager::instance()->loadPluginByPath(getPluginPath() + "Release");
#else
			PluginManager::instance()->loadPluginByPath(getPluginPath() + "Debug");
#endif // DEBUG
		}

        mMainWindow = std::make_shared<PMainWindow>(this);
        mMainWindow->resize(width, height);
    }

    void QtApp::mainLoop()
    {
        QFile file(":/dyno/DarkStyle.qss");
        //QFile file(":/dyno/DarkStyle.qss");
        file.open(QIODevice::ReadOnly);

        QString style = file.readAll();
        mQApp->setStyleSheet(style);

        mMainWindow->show();
        mQApp->exec();
    }

	void QtApp::setSceneGraph(std::shared_ptr<SceneGraph> scn)
	{
        AppBase::setSceneGraph(scn);
        SceneGraphFactory::instance()->pushScene(scn);
	}

    RenderWindow* QtApp::renderWindow()
	{
        return dynamic_cast<RenderWindow*>(mMainWindow->openglWidget());
	}

}
