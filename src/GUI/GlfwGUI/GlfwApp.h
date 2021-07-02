#pragma once
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>

#include <glad/glad.h>
// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

#include "AppBase.h"

namespace dyno {

	class Camera;
	class RenderEngine;
	class RenderTarget;
	class RenderParams;

	enum ButtonState
	{
		GLFW_DOWN = 0,
		GLFW_UP
	};

	enum CameraType
	{
		Orbit,
		TrackBall
	};

    class GlfwApp : public AppBase
    {
    public:
        GlfwApp(int argc = 0, char **argv = NULL);
		GlfwApp(int width, int height);
        ~GlfwApp();

        void createWindow(int width, int height) override;
		void setupCamera();
        void mainLoop() override;

		void setCameraType(CameraType type);
		CameraType cameraType() { return mCameraType; }

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

		std::shared_ptr<Camera> activeCamera() { return mCamera; }

		//save screenshot to file
		bool saveScreen(const std::string &file_name) const;  //save to file with given name
		bool saveScreen();                                    //save to file with default name "screen_capture_XXX.png"

		void enableSaveScreen() { mSaveScreenToggle = true; }
		void disableSaveScreen() { mSaveScreenToggle = false; };
		void setOutputPath(std::string path) { mOutputPath = path; }
		void setSaveScreenInterval(int n) { mSaveScreenInterval = n < 1 ? 1 : n; }
		int getSaveScreenInternal() { return mSaveScreenInterval; }

		void toggleAnimation();

		int getWidth();
		int getHeight();

		bool getCameraRotateFlag();

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
		
		// The static bool variable is used for avoiding Camera Rotating in Imgui Windows
		static bool mOpenCameraRotate;

		int mSaveScreenInterval = 1;

		//current screen capture file index
		uint mSaveScreenIndex;

		std::string mOutputPath;
		std::string mWindowTitle;

		Vec4f mClearColor = Vec4f(0.45f, 0.55f, 0.60f, 1.00f);

		std::shared_ptr<Camera> mCamera;
		CameraType mCameraType = CameraType::Orbit;

		RenderEngine* mRenderEngine;
		RenderTarget* mRenderTarget;
		RenderParams* mRenderParams;

	public:
		bool			mUseNewRenderEngine = true;
		
    };

}