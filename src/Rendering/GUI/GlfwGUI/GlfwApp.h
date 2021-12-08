#pragma once

#include <Platform.h>
#include <ImWindow.h>

#include "AppBase.h"

struct GLFWwindow;
namespace dyno {

	class Camera;
	struct Picture;

	enum ButtonState
	{
		GLFW_DOWN = 0,
		GLFW_UP
	};

    class GlfwApp : public AppBase
    {
    public:
        GlfwApp(int argc = 0, char **argv = NULL);
		GlfwApp(int width, int height);
        ~GlfwApp();

        void createWindow(int width, int height) override;

        void mainLoop() override;

		const std::string& name() const;

		void setCursorPos(double x, double y);
		double getCursorPosX();
		double getCursorPosY();
		
		void setWindowSize(int width, int height);

		void setButtonType(uint button) { mButtonType = button; }
		void setButtonMode(uint mode) { mButtonMode = mode; }
		void setButtonAction(uint action) { mButtonAction = action; }
		void setButtonState(ButtonState state) { mButtonState = state; }

		uint getButtonType() const { return mButtonType; }
		uint getButtonMode() { return mButtonMode; }
		uint getButtonAction() const { return mButtonAction; }
		ButtonState getButtonState() const { return mButtonState; }

		std::shared_ptr<Camera> activeCamera();

		//save screenshot to file
		bool saveScreen(const std::string &file_name) const;  //save to file with given name
		bool saveScreen();                                    //save to file with default name "screen_capture_XXX.png"

		void enableSaveScreen() { mSaveScreenToggle = true; }
		void disableSaveScreen() { mSaveScreenToggle = false; };
		void setOutputPath(std::string path) { mOutputPath = path; }
		void setSaveScreenInterval(int n) { mSaveScreenInterval = n < 1 ? 1 : n; }
		int getSaveScreenInternal() { return mSaveScreenInterval; }

		void toggleAnimation();
		void toggleImGUI();

		int getWidth();
		int getHeight();

		// ImGui extend function
		// 全局样式设定
		void initializeStyle();

		RenderEngine* renderEngine() { return mRenderEngine; }


	protected:
		void initCallbacks();    //init default callbacks

		void drawScene(void);

		static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
		static void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void reshapeCallback(GLFWwindow* window, int w, int h);
		static void cursorPosCallback(GLFWwindow* window, double x, double y);
		static void cursorEnterCallback(GLFWwindow* window, int entered);
		static void scrollCallback(GLFWwindow* window, double offsetX, double OffsetY);


		
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
		bool mSaveScreenToggle = false;
		bool mBackgroundToggle = true;
		bool mBoundingboxToggle = false;

		int mSaveScreenInterval = 1;

		//current screen capture file index
		uint mSaveScreenIndex;

		std::string mOutputPath;
		std::string mWindowTitle;

	private:
		bool mShowImWindow = true;

		ImWindow mImWindow;
    };

}