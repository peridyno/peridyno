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

        void createWindow(int width, int height) override;
        void mainLoop() override;

        void setRenderEngine(std::shared_ptr<RenderEngine> engine) override;
        void setSceneGraph(std::shared_ptr<SceneGraph> scn);

        std::shared_ptr<RenderEngine> renderEngine() override;

    private:
        std::shared_ptr<QApplication> m_app;
        std::shared_ptr<PMainWindow> m_mainWindow;
    };

}