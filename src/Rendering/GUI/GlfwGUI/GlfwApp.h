#pragma once

#include <Platform.h>
#include <RenderWindow.h>
#include <ImWindow.h>

#include "AppBase.h"

struct GLFWwindow;
namespace dyno {

	class Camera;
	class Node;
	struct Picture;

    class GlfwApp : public AppBase
    {
    public:
        GlfwApp(int argc = 0, char **argv = NULL);
        ~GlfwApp();

        void createWindow(int width, int height, bool usePlugin = false) override;
        void createWindow(int width, int height, std::shared_ptr<RenderEngine> engine);

        void resize(int width, int height);

        std::shared_ptr<RenderWindow> renderWindow() { return mRenderWindow; }

        void mainLoop() override;

	private:
		std::shared_ptr<RenderWindow> mRenderWindow;
    };

}