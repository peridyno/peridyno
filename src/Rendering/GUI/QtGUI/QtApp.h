#pragma once
#include <memory>
#include "AppBase.h"

#include <QApplication>

namespace dyno {

	class PMainWindow;
    class SceneGraph;

    class QtApp : public AppBase
    {
    public:
        QtApp(int argc = 0, char **argv = NULL);
        ~QtApp();

        void createWindow(int width, int height, bool usePlugin = true) override;
        void mainLoop() override;

        void setSceneGraph(std::shared_ptr<SceneGraph> scn);

    private:
        std::shared_ptr<QApplication> m_app;
        std::shared_ptr<PMainWindow> m_mainWindow;
    };

}