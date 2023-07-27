#pragma once
#include <memory>
#include "AppBase.h"

#include <QApplication>
#include <RenderWindow.h>

namespace dyno {

	class PMainWindow;
    class SceneGraph;

    class QtApp : public AppBase
    {
    public:
        QtApp(int argc = 0, char **argv = NULL);
        ~QtApp();

        void initialize(int width, int height, bool usePlugin = true) override;
        void mainLoop() override;

        void setSceneGraph(std::shared_ptr<SceneGraph> scn);

        /**
         * Set customized window title; otherwise, the default title will be shown
         */
        void setWindowTitle(const std::string& str);

        RenderWindow* renderWindow();

    private:
        std::shared_ptr<QApplication> mQApp;
        std::shared_ptr<PMainWindow> mMainWindow;
    };

}