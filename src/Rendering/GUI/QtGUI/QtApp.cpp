#include <QMap>
#include <QDebug>
#include <QFile>
#include <QApplication>

#include "QtApp.h"
#include "PMainWindow.h"
#include "POpenGLWidget.h"
#include "Log.h"

#include "SceneGraphFactory.h"
#include "Plugin/PluginManager.h"

#include <OrbitCamera.h>

namespace dyno {
    QtApp::QtApp(int argc, char **argv)
        : AppBase()
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

        //To fix the displace scaling issue
        QApplication::setAttribute(Qt::AA_Use96Dpi);
        qputenv("QT_ENABLE_HIGHDPI_SCALING", "0");

        mQApp = std::make_shared<QApplication>(argc, argv);

        //Set default GUI style
		QFile file(":/dyno/DarkStyle.qss");
		file.open(QIODevice::ReadOnly);

		QString style = file.readAll();
		mQApp->setStyleSheet(style);

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
        mMainWindow->show();
    }

    void QtApp::mainLoop()
    {       
        mQApp->exec();
    }

	void QtApp::setSceneGraph(std::shared_ptr<SceneGraph> scn)
	{
        AppBase::setSceneGraph(scn);
        SceneGraphFactory::instance()->pushScene(scn);
	}

	void QtApp::setWindowTitle(const std::string& str)
	{
        mMainWindow->setWindowTitle(QString::fromStdString(str));
	}

	RenderWindow* QtApp::renderWindow()
	{
        return dynamic_cast<RenderWindow*>(mMainWindow->openglWidget());
	}

}
 