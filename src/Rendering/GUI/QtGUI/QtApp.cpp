#include <QMap>
#include <QDebug>
#include <QFile>
#include "QtApp.h"
#include "PMainWindow.h"
#include "Log.h"

#include "SceneGraphFactory.h"
#include "Plugin/PluginManager.h"

#include <OrbitCamera.h>

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

    void QtApp::createWindow(int width, int height, bool usePlugin)
    {
		if (usePlugin)
		{
#ifdef NDEBUG
			PluginManager::instance()->loadPluginByPath(getPluginPath() + "Release");
#else
			PluginManager::instance()->loadPluginByPath(getPluginPath() + "Debug");
#endif // DEBUG
		}

        m_mainWindow = std::make_shared<PMainWindow>(this);
        m_mainWindow->resize(width, height);
    }

    void QtApp::mainLoop()
    {
        QFile file(":/dyno/DarkStyle.qss");
        //QFile file(":/dyno/DarkStyle.qss");
        file.open(QIODevice::ReadOnly);

        QString style = file.readAll();
        m_app->setStyleSheet(style);

        m_mainWindow->show();
        m_app->exec();
    }

	void QtApp::setSceneGraph(std::shared_ptr<SceneGraph> scn)
	{
        AppBase::setSceneGraph(scn);
        SceneGraphFactory::instance()->pushScene(scn);
	}

}
