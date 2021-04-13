#pragma once
#include "AppBase.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>

#include <glad/gl.h>
// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

namespace dyno {

    class GlfwApp : public AppBase
    {
    public:
        GlfwApp(int argc = 0, char **argv = NULL);
		GlfwApp(int width, int height);
        ~GlfwApp();

        void createWindow(int width, int height) override;
        void mainLoop() override;

	protected:
		void initCallbacks();    //init default callbacks

		void drawScene(void);
		void drawBackground();
		void drawAxis();

		static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
		static void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    private:
		//pointers to callback methods
		void(*mMouseButtonCallback)(GLFWwindow* window, int button, int action, int mods);
		void(*mKeyboardCallback)(GLFWwindow* window, int key, int scancode, int action, int mods);

		GLFWwindow* window = nullptr;

		unsigned int mWidth;
		unsigned int mHeight;

		ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

		GLfloat alpha = 210.f, beta = -70.f;
		GLfloat zoom = 2.f;
    };

}