#include "GlfwRenderWindow.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include "SceneGraph.h"
#include "Module/MouseInputModule.h"
#include "Log.h"

#include <RenderEngine.h>
#include <OrbitCamera.h>
#include <TrackballCamera.h>

#include <GLRenderEngine.h>
#include <SceneGraphFactory.h>

#include <glad/glad.h>
// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "ImGuizmo.h"

#include <ImWidget.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

namespace dyno 
{
	static void RecieveLogMessage(const Log::Message& m)
	{
		switch (m.type)
		{
		case Log::Info:
			std::cout << ">>>: " << m.text << std::endl; break;
		case Log::Warning:
			std::cout << "???: " << m.text << std::endl; break;
		case Log::Error:
			std::cout << "!!!: " << m.text << std::endl; break;
		case Log::User:
			std::cout << ">>>: " << m.text << std::endl; break;
		default: break;
		}
	}

	static void glfw_error_callback(int error, const char* description)
	{
		fprintf(stderr, "Glfw Error %d: %s\n", error, description);
	}

	GlfwRenderWindow::GlfwRenderWindow(int argc /*= 0*/, char **argv /*= NULL*/)
		: RenderWindow()
	{
		Log::setUserReceiver(&RecieveLogMessage);

		// create render engine
		mRenderEngine = std::make_shared<GLRenderEngine>();
	}

	GlfwRenderWindow::~GlfwRenderWindow()
	{
		// Cleanup
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		glfwDestroyWindow(mWindow);
		glfwTerminate();
	}

	void GlfwRenderWindow::initialize(int width, int height)
	{
		mWindowTitle = std::string("PeriDyno ") + std::to_string(PERIDYNO_VERSION_MAJOR) + std::string(".") + std::to_string(PERIDYNO_VERSION_MINOR) + std::string(".") + std::to_string(PERIDYNO_VERSION_PATCH);

		// Setup window
		glfwSetErrorCallback(glfw_error_callback);
		if (!glfwInit())
			return;

		// Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
	// GL ES 2.0 + GLSL 100
		const char* glsl_version = "#version 100";
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
	// GL 3.2 + GLSL 150
		const char* glsl_version = "#version 150";
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
	// GL 3.0 + GLSL 130
		const char* glsl_version = "#version 130";
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
		//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif
		// enable multisamples for anti alias
		glfwWindowHint(GLFW_SAMPLES, 4);

		// Create window with graphics context
		mWindow = glfwCreateWindow(width, height, mWindowTitle.c_str(), NULL, NULL);
		if (mWindow == NULL)
			return;

		initCallbacks();
		
		glfwMakeContextCurrent(mWindow);
		
		if (!gladLoadGL()) {
			Log::sendMessage(Log::Error, "Failed to load GLAD!");
			//SPDLOG_CRITICAL("Failed to load GLAD!");
			exit(-1);
		}

		glfwSwapInterval(1); // Enable vsync

		glfwSetWindowUserPointer(mWindow, this);

		// Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
		bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
		bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
		bool err = gladLoadGL() == 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD2)
		bool err = gladLoadGL(glfwGetProcAddress) == 0; // glad2 recommend using the windowing library loader instead of the (optionally) bundled one.
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
		bool err = false;
		glbinding::Binding::initialize();
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
		bool err = false;
		glbinding::initialize([](const char* name) { return (glbinding::ProcAddress)glfwGetProcAddress(name); });
#else
		bool err = false; // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader is likely to requires some form of initialization.
#endif
		if (err)
		{
			fprintf(stderr, "Failed to initialize OpenGL loader!\n");
			return;
		}

		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
		//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();
		//ImGui::StyleColorsClassic();
		// initializeStyle();

		// Setup Platform/Renderer backends
		ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
		ImGui_ImplOpenGL3_Init(glsl_version);

		// Get Context scale
		float xscale, yscale;
		glfwGetMonitorContentScale(glfwGetPrimaryMonitor(), &xscale, &yscale);

		// initialize rendering engine
		mRenderEngine->initialize();

		// Jian: initialize ImWindow
		mImWindow.initialize(xscale);

		this->setWindowSize(width, height);
	}

	void GlfwRenderWindow::initializeStyle()
	{
		ImGuiStyle& style = ImGui::GetStyle();
		style.WindowRounding = 6.0f;
		style.ChildRounding = 6.0f;
		style.FrameRounding = 6.0f;
		style.PopupRounding = 6.0f;
	}
	
	void GlfwRenderWindow::mainLoop()
	{
		auto activeScene = SceneGraphFactory::instance()->active();

		activeScene->reset();

		// Main loop
		while (!glfwWindowShouldClose(mWindow))
		{
			glfwPollEvents();

			if (mAnimationToggle){

				if (this->isScreenRecordingOn())
				{
					saveScreen(activeScene->getFrameNumber());
				}

				activeScene->takeOneFrame();
			}
			
			activeScene->updateGraphicsContext();

			// update rendering params
			mRenderParams.width = mCamera->viewportWidth();
			mRenderParams.height = mCamera->viewportHeight();

			// check if window is minimized
			if (mRenderParams.width == 0 || mRenderParams.height == 0)
				continue;

			mRenderParams.transforms.model = glm::mat4(1);	 // TODO: world transform?
			mRenderParams.transforms.view = mCamera->getViewMat();
			mRenderParams.transforms.proj = mCamera->getProjMat();
			mRenderParams.unitScale = mCamera->unitScale();

			mRenderEngine->draw(activeScene.get(), mRenderParams);

			// Start the Dear ImGui frame
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			if(showImGUI())
				mImWindow.draw(this);

// 			// Draw widgets
// 			// TODO: maybe move into mImWindow...
// 			for (auto widget : mWidgets)
// 			{
// 				widget->update();
// 				widget->paint();
// 			}



// 			// draw a pick rect
// 			if (mButtonType == GLFW_MOUSE_BUTTON_LEFT &&
// 				mButtonAction == GLFW_PRESS &&
// 				mButtonMode == 0 && 
// 				!ImGuizmo::IsUsing() &&
// 				!ImGui::GetIO().WantCaptureMouse) {
// 				double xpos, ypos;
// 				glfwGetCursorPos(mWindow, &xpos, &ypos);
// 
// 				ImVec2 pMin = { fminf(xpos, mCursorPosX), fminf(ypos, mCursorPosY) };
// 				ImVec2 pMax = { fmaxf(xpos, mCursorPosX), fmaxf(ypos, mCursorPosY) };			
// 
// 				// visible rectangle
// 				if (pMin.x != pMax.x || pMin.y != pMax.y) {
// 					// fill
// 					ImGui::GetBackgroundDrawList()->AddRectFilled(pMin, pMax, ImColor{ 0.2f, 0.2f, 0.2f, 0.5f });
// 					// border
// 					ImGui::GetBackgroundDrawList()->AddRect(pMin, pMax, ImColor{ 0.8f, 0.8f, 0.8f, 0.8f }, 0, 0, 1.5f);
// 				}
// 			}

			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

			glfwSwapBuffers(mWindow);
		}

		mRenderEngine->terminate();
	}

	const std::string& GlfwRenderWindow::name() const
	{
		return mWindowTitle;
	}

	void GlfwRenderWindow::setWindowTitle(const std::string& title)
	{
		glfwSetWindowTitle(mWindow, title.c_str());
	}

	void GlfwRenderWindow::setCursorPos(double x, double y)
	{
		mCursorPosX = x;
		mCursorPosY = y;
	}

	double GlfwRenderWindow::getCursorPosX()
	{
		return mCursorPosX;
	}

	double GlfwRenderWindow::getCursorPosY()
	{
		return mCursorPosY;
	}

	void GlfwRenderWindow::onSaveScreen(const std::string &filename)
	{
		int width;
		int height;
		glfwGetFramebufferSize(mWindow, &width, &height);

		unsigned char *data = new unsigned char[width * height * 3];  //RGB
		assert(data);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0); 
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, (void*)data);

		stbi_flip_vertically_on_write(true);
		int status = stbi_write_bmp(filename.c_str(), width, height, 3, data);

		delete data;
	}

	void GlfwRenderWindow::turnOnVSync()
	{
		glfwSwapInterval(1);
	}

	void GlfwRenderWindow::turnOffVSync()
	{
		glfwSwapInterval(0);
	}

	void GlfwRenderWindow::toggleAnimation()
	{
		mAnimationToggle = !mAnimationToggle;
	}

	int GlfwRenderWindow::getWidth()
	{
		return getCamera()->viewportWidth();
	}

	int GlfwRenderWindow::getHeight()
	{
		return getCamera()->viewportHeight();
	}

	void GlfwRenderWindow::initCallbacks()
	{
		mMouseButtonFunc = GlfwRenderWindow::mouseButtonCallback;
		mKeyboardFunc = GlfwRenderWindow::keyboardCallback;
		mReshapeFunc = GlfwRenderWindow::reshapeCallback;
		mCursorPosFunc = GlfwRenderWindow::cursorPosCallback;
		mCursorEnterFunc = GlfwRenderWindow::cursorEnterCallback;
		mScrollFunc = GlfwRenderWindow::scrollCallback;

		glfwSetMouseButtonCallback(mWindow, mMouseButtonFunc);
		glfwSetKeyCallback(mWindow, mKeyboardFunc);
		glfwSetFramebufferSizeCallback(mWindow, mReshapeFunc);
		glfwSetCursorPosCallback(mWindow, mCursorPosFunc);
		glfwSetCursorEnterCallback(mWindow, mCursorEnterFunc);
		glfwSetScrollCallback(mWindow, mScrollFunc);

	}

	void GlfwRenderWindow::drawScene(void)
	{
		
	}

	void GlfwRenderWindow::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
	{
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		GlfwRenderWindow* activeWindow = (GlfwRenderWindow*)glfwGetWindowUserPointer(window);

		// Object selection
		if (activeWindow->getSelectionMode() == RenderWindow::OBJECT_MODE)
		{
			// handle picking
			if (activeWindow->getButtonType() == GLFW_MOUSE_BUTTON_LEFT &&
				activeWindow->getButtonAction() == GLFW_PRESS &&
				activeWindow->getButtonMode() == 0 &&
				action == GLFW_RELEASE) {

				// in picking
				int x = fmin(xpos, activeWindow->getCursorPosX());
				int y = fmax(ypos, activeWindow->getCursorPosY());
				int w = fabs(xpos - activeWindow->getCursorPosX());
				int h = fabs(ypos - activeWindow->getCursorPosY());
				// flip y to texture space...
				y = activeWindow->getHeight() - y - 1;

				const auto& selection = activeWindow->select(x, y, w, h);
			}
		}

		auto camera = activeWindow->getCamera();

		activeWindow->setButtonType(button);
		activeWindow->setButtonAction(action);
		activeWindow->setButtonMode(mods);

		// Primitive selection
		if (activeWindow->getSelectionMode() == RenderWindow::PRIMITIVE_MODE)
		{
			PMouseEvent mouseEvent;
			mouseEvent.ray = camera->castRayInWorldSpace((float)xpos, (float)ypos);
			mouseEvent.buttonType = (PButtonType)button;
			mouseEvent.actionType = (PActionType)action;
			mouseEvent.mods = (PModifierBits)activeWindow->getButtonMode();
			mouseEvent.camera = camera;
			mouseEvent.x = (float)xpos;
			mouseEvent.y = (float)ypos;

			auto activeScene = SceneGraphFactory::instance()->active();

			if (activeScene->getWorkMode() == SceneGraph::EDIT_MODE)
			{
				activeScene->onMouseEvent(mouseEvent, activeWindow->getCurrentSelectedNode());
			}
			else
				activeScene->onMouseEvent(mouseEvent);
			

			if (action == GLFW_PRESS)
			{
				activeWindow->imWindow()->mousePressEvent(mouseEvent);
			}

			if (action == GLFW_RELEASE)
			{
				activeWindow->imWindow()->mouseReleaseEvent(mouseEvent);
			}
		}

		if (action == GLFW_PRESS)
		{
			// if(mOpenCameraRotate)
			camera->registerPoint((float)xpos, (float)ypos);
			activeWindow->setButtonState(GLFW_DOWN);
		}
		else
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		if (action == GLFW_RELEASE)
		{
			activeWindow->setButtonState(GLFW_UP);
		}

		// update cursor position record
		
		if (action == GLFW_PRESS) 
		{
			activeWindow->setCursorPos(xpos, ypos);
			activeWindow->mCursorTempX = xpos;
		}
		else 
		{
			activeWindow->setCursorPos(-1, -1);
			activeWindow->mCursorTempX = -1;
		}

	}

	void GlfwRenderWindow::cursorPosCallback(GLFWwindow* window, double x, double y)
	{
		GlfwRenderWindow* activeWindow = (GlfwRenderWindow*)glfwGetWindowUserPointer(window); // User Pointer
		auto camera = activeWindow->getCamera();

		PMouseEvent mouseEvent;
		mouseEvent.ray = camera->castRayInWorldSpace((float)x, (float)y);
		mouseEvent.buttonType = (PButtonType)activeWindow->getButtonType();
		mouseEvent.actionType = PActionType::AT_REPEAT;
		mouseEvent.mods = (PModifierBits)activeWindow->getButtonMode();
		mouseEvent.camera = camera;
		mouseEvent.x = (float)x;
		mouseEvent.y = (float)y;

		auto activeScene = SceneGraphFactory::instance()->active();
		activeScene->onMouseEvent(mouseEvent);

		if (activeWindow->getButtonType() == GLFW_MOUSE_BUTTON_LEFT &&
			activeWindow->getButtonState() == GLFW_DOWN &&
			activeWindow->getButtonMode() == GLFW_MOD_ALT &&
			!activeWindow->mImWindow.cameraLocked()) 
		{
			camera->rotateToPoint(x, y);
		}
		else if (
			activeWindow->getButtonType() == GLFW_MOUSE_BUTTON_MIDDLE &&
			activeWindow->getButtonState() == GLFW_DOWN && 
			activeWindow->getButtonMode() == GLFW_MOD_ALT &&
			!activeWindow->mImWindow.cameraLocked()) 
		{
			camera->translateToPoint(x, y);
		}
		else if (
			activeWindow->getButtonType() == GLFW_MOUSE_BUTTON_RIGHT &&
			activeWindow->getButtonState() == GLFW_DOWN &&
			activeWindow->getButtonMode() == GLFW_MOD_ALT &&
			!activeWindow->mImWindow.cameraLocked())////
		{
			if (activeWindow->mCursorTempX != -1) 
			{
				camera->zoom(-0.005 * (x - activeWindow->mCursorTempX));
				activeWindow->mCursorTempX = x;
			}
		}
		activeWindow->imWindow()->mouseMoveEvent(mouseEvent); 
	}

	void GlfwRenderWindow::cursorEnterCallback(GLFWwindow* window, int entered)
	{
		if (entered)
		{
			// The cursor entered the content area of the window
		}
		else
		{
			// The cursor left the content area of the window
		}
	}

	void GlfwRenderWindow::scrollCallback(GLFWwindow* window, double offsetX, double OffsetY)
	{
		GlfwRenderWindow* activeWindow = (GlfwRenderWindow*)glfwGetWindowUserPointer(window);
		auto camera = activeWindow->getCamera();

		if (!activeWindow->mImWindow.cameraLocked())
		{
			int state = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL);
			int altState = glfwGetKey(window, GLFW_KEY_LEFT_ALT);
			//If the left control key is pressed, slow the zoom speed. 
			if (state == GLFW_PRESS && altState == GLFW_PRESS)
				camera->zoom(-0.1*OffsetY);
			else if (altState == GLFW_PRESS)
				camera->zoom(-OffsetY);
		}
	}

	void GlfwRenderWindow::keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		GlfwRenderWindow* activeWindow = (GlfwRenderWindow*)glfwGetWindowUserPointer(window);

		PKeyboardEvent keyEvent;
		keyEvent.key = (PKeyboardType)key;
		keyEvent.action = (PActionType)action;
		keyEvent.mods = (PModifierBits)mods;

		auto activeScene = SceneGraphFactory::instance()->active();
		activeScene->onKeyboardEvent(keyEvent);

		if (action != GLFW_PRESS)
			return;

		switch (key)
		{
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GLFW_TRUE);
			break;
		case GLFW_KEY_SPACE:
			activeWindow->toggleAnimation();
			break;
		case GLFW_KEY_LEFT:
			break;
		case GLFW_KEY_RIGHT:
			break;
		case GLFW_KEY_UP:
			break;
		case GLFW_KEY_DOWN:
			break;
		case GLFW_KEY_PAGE_UP:
			break;
		case GLFW_KEY_PAGE_DOWN:
			break;
		case GLFW_KEY_N:
			activeScene->takeOneFrame();
			activeScene->updateGraphicsContext();
			break;
		case GLFW_KEY_F1:
			activeWindow->toggleImGUI();
			break;
		default:
			break;
		}
	}

	void GlfwRenderWindow::reshapeCallback(GLFWwindow* window, int w, int h)
	{
		GlfwRenderWindow* app = (GlfwRenderWindow*)glfwGetWindowUserPointer(window);
		app->setWindowSize(w, h);
	}

}