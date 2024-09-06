#pragma once

#include <Platform.h>
#include <RenderWindow.h>
#include <ImWindow.h>

#include "AppBase.h"
#include "GlfwRenderWindow.h"

namespace dyno 
{
    class GlfwApp : public AppBase
    {
    public:
        GlfwApp(int argc = 0, char **argv = NULL);
        ~GlfwApp();

        void initialize(int width, int height, bool usePlugin = false) override;

        std::shared_ptr<RenderWindow> renderWindow() { return mRenderWindow; }

        void mainLoop() override;

        void setWindowTitle(const std::string& title);

	private:
		std::shared_ptr<GlfwRenderWindow> mRenderWindow;
    };
}