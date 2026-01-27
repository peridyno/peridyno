#pragma once

#include <Platform.h>
#include <ImWindow.h>

#include "RenderWindow.h"

struct GLFWwindow;
namespace dyno {

	class Camera;
	class Node;
	struct Picture;

	enum ButtonState
	{
		GLFW_DOWN = 0,
		GLFW_UP
	};

    class GlfwRenderWindow : public RenderWindow
    {
    public:
		GlfwRenderWindow(int argc = 0, char **argv = NULL);
        ~GlfwRenderWindow();

        void initialize(int width, int height) override;

        void mainLoop() override;

		const std::string& name() const;

		void setWindowTitle(const std::string& title);

		void setCursorPos(double x, double y);
		double getCursorPosX();
		double getCursorPosY();
		
		void setButtonType(uint button) { mButtonType = button; }
		void setButtonMode(uint mode) { mButtonMode = mode; }
		void setButtonAction(uint action) { mButtonAction = action; }
		void setButtonState(ButtonState state) { mButtonState = state; }

		void setDefaultAnimationOption(bool op) override { mAnimationToggle = op; }

		uint getButtonType() const { return mButtonType; }
		uint getButtonMode() { return mButtonMode; }
		uint getButtonAction() const { return mButtonAction; }
		ButtonState getButtonState() const { return mButtonState; }

		void turnOnVSync();
		void turnOffVSync();

		void toggleAnimation();

		int getWidth();
		int getHeight();

		// ImGui extend function
		// 全局样式设定
		void initializeStyle();

		ImWindow* imWindow() { return &mImWindow; }

	protected:
		void initCallbacks();    //init default callbacks

		void drawScene(void);

		static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
		static void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void reshapeCallback(GLFWwindow* window, int w, int h);
		static void cursorPosCallback(GLFWwindow* window, double x, double y);
		static void cursorEnterCallback(GLFWwindow* window, int entered);
		static void scrollCallback(GLFWwindow* window, double offsetX, double OffsetY);

		//save screenshot to file
		void onSaveScreen(const std::string& filename) override;  //save to file with given name
		
    private:
		//pointers to callback methods
		void(*mMouseButtonFunc)(GLFWwindow* window, int button, int action, int mods);
		void(*mKeyboardFunc)(GLFWwindow* window, int key, int scancode, int action, int mods);
		void(*mReshapeFunc)(GLFWwindow* window, int w, int h);
		void(*mCursorPosFunc)(GLFWwindow* window, double x, double y);
		void(*mCursorEnterFunc)(GLFWwindow* window, int entered);
		void(*mScrollFunc)(GLFWwindow* window, double offsetX, double OffsetY);

		GLFWwindow* mWindow = nullptr;

		//state of the mouse
		uint mButtonType;
		uint mButtonMode;
		uint mButtonAction;
		ButtonState mButtonState;

		int mPlaneSize = 4;

		double mCursorPosX;
		double mCursorPosY;

		bool mAnimationToggle = false;

		bool mBackgroundToggle = true;
		bool mBoundingboxToggle = false;

		std::string mWindowTitle;

	private:
		ImWindow mImWindow;
    };

}