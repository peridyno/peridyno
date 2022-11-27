#include "GlfwApp.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#ifdef CUDA_BACKEND
#include "Image_IO/image_io.h"
#endif // CUAD_BACKEND

#include "SceneGraph.h"
#include "Module/MouseInputModule.h"
#include "Log.h"

#include <Rendering.h>
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

	GlfwApp::GlfwApp(int argc /*= 0*/, char **argv /*= NULL*/)
	{
		Log::setUserReceiver(&RecieveLogMessage);

		// create render engine
		mRenderEngine = std::make_shared<GLRenderEngine>();

		// create a default camera
		mCamera = std::make_shared<OrbitCamera>();
		mCamera->setWidth(64);
		mCamera->setHeight(64);
		mCamera->registerPoint(0, 0);
		mCamera->rotateToPoint(-32, 12);
	}

	GlfwApp::~GlfwApp()
	{
		// Cleanup
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		glfwDestroyWindow(mWindow);
		glfwTerminate();
	}

	void GlfwApp::createWindow(int width, int height, bool usePlugin)
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
		mRenderEngine->initialize(width, height);

		// Jian: initialize ImWindow
		mImWindow.initialize(xscale);

		this->setWindowSize(width, height);
	}

	void GlfwApp::initializeStyle()
	{
		ImGuiStyle& style = ImGui::GetStyle();
		style.WindowRounding = 6.0f;
		style.ChildRounding = 6.0f;
		style.FrameRounding = 6.0f;
		style.PopupRounding = 6.0f;
	}
	
	void GlfwApp::setSceneGraph(std::shared_ptr<SceneGraph> scn)
	{
		AppBase::setSceneGraph(scn);
		SceneGraphFactory::instance()->pushScene(scn);
	}

	void GlfwApp::mainLoop()
	{
		auto activeScene = SceneGraphFactory::instance()->active();

		activeScene->reset();

		// Main loop
		while (!glfwWindowShouldClose(mWindow))
		{
			
			glfwPollEvents();

			if (mAnimationToggle){

				if (mSaveScreenToggle)
				{
					if (activeScene->getFrameNumber() % mSaveScreenInterval == 0)
						saveScreen();
				}

				activeScene->takeOneFrame();
			}
			
			activeScene->updateGraphicsContext();
				
			mRenderParams.proj = mCamera->getProjMat();
			mRenderParams.view = mCamera->getViewMat();

			// Jian SHI: hack for unit scaling...
			float planeScale = mRenderParams.planeScale;
			float rulerScale = mRenderParams.rulerScale;
			mRenderParams.planeScale *= mCamera->distanceUnit();
			mRenderParams.rulerScale *= mCamera->distanceUnit();

			mRenderEngine->draw(activeScene.get(), mRenderParams);

			mRenderParams.planeScale = planeScale;
			mRenderParams.rulerScale = rulerScale;

			// Start the Dear ImGui frame
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			if(mShowImWindow)
				mImWindow.draw(this);

// 			// Draw widgets
// 			// TODO: maybe move into mImWindow...
// 			for (auto widget : mWidgets)
// 			{
// 				widget->update();
// 				widget->paint();
// 			}

			if (currNode) {

				auto view = getRenderParams().view;
				auto proj = getRenderParams().proj;

				ImGuiIO& io = ImGui::GetIO();
				ImGuizmo::BeginFrame();
				ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

				// TODO: node transform interface?
				glm::mat4 transform(1.f);

				auto bbox = currNode->boundingBox();
				auto center = (bbox.lower + bbox.upper) * 0.5f;
				transform[3][0] = center[0];
				transform[3][1] = center[1];
				transform[3][2] = center[2];

				if (ImGuizmo::Manipulate(&view[0][0], &proj[0][0], ImGuizmo::TRANSLATE, ImGuizmo::WORLD, &transform[0][0], NULL, NULL, NULL, NULL))
				{
					// TODO: apply transform
				}
			}

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
	}

	const std::string& GlfwApp::name() const
	{
		return mWindowTitle;
	}

	void GlfwApp::setCursorPos(double x, double y)
	{
		mCursorPosX = x;
		mCursorPosY = y;
	}

	double GlfwApp::getCursorPosX()
	{
		return mCursorPosX;
	}

	double GlfwApp::getCursorPosY()
	{
		return mCursorPosY;
	}



	bool GlfwApp::saveScreen(const std::string &file_name) const
	{
		int width;
		int height;
		glfwGetFramebufferSize(mWindow, &width, &height);

		unsigned char *data = new unsigned char[width * height * 3];  //RGB
		assert(data);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0); 
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, (void*)data);

#ifdef CUDA_BACKEND
		Image image(width, height, Image::RGB, data);
		image.flipVertically();
		bool status = ImageIO::save(file_name, &image);
		delete[] data;
		return status;
#else
		return false;
#endif
		
	}

	bool GlfwApp::saveScreen()
	{
		std::stringstream adaptor;
		adaptor << mSaveScreenIndex++;
		std::string index_str;
		adaptor >> index_str;
		std::string file_name = mOutputPath + std::string("screen_capture_") + index_str + std::string(".ppm");
		return saveScreen(file_name);
	}


	void GlfwApp::turnOnVSync()
	{
		glfwSwapInterval(1);
	}

	void GlfwApp::turnOffVSync()
	{
		glfwSwapInterval(0);
	}

	void GlfwApp::toggleAnimation()
	{
		mAnimationToggle = !mAnimationToggle;
	}

	void GlfwApp::toggleImGUI()
	{
		mShowImWindow = !mShowImWindow;
	}

	int GlfwApp::getWidth()
	{
		return getCamera()->viewportWidth();
	}

	int GlfwApp::getHeight()
	{
		return getCamera()->viewportHeight();
	}

	void GlfwApp::initCallbacks()
	{
		mMouseButtonFunc = GlfwApp::mouseButtonCallback;
		mKeyboardFunc = GlfwApp::keyboardCallback;
		mReshapeFunc = GlfwApp::reshapeCallback;
		mCursorPosFunc = GlfwApp::cursorPosCallback;
		mCursorEnterFunc = GlfwApp::cursorEnterCallback;
		mScrollFunc = GlfwApp::scrollCallback;

		glfwSetMouseButtonCallback(mWindow, mMouseButtonFunc);
		glfwSetKeyCallback(mWindow, mKeyboardFunc);
		glfwSetFramebufferSizeCallback(mWindow, mReshapeFunc);
		glfwSetCursorPosCallback(mWindow, mCursorPosFunc);
		glfwSetCursorEnterCallback(mWindow, mCursorEnterFunc);
		glfwSetScrollCallback(mWindow, mScrollFunc);
	}

	void GlfwApp::drawScene(void)
	{
		
	}

	void GlfwApp::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
	{
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		GlfwApp* activeWindow = (GlfwApp*)glfwGetWindowUserPointer(window);

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

			auto items = activeWindow->mRenderEngine->select(x, y, w, h);
			// print selected result...
			printf("Picking: (%d, %d) - (%d, %d), %d items...\n", x, y, w, h, items.size());

			if (!items.empty()) {
				// pick the last one?
				auto node = items[0].node;
				int instance = items[0].instance;

				auto bbox = node->boundingBox();

				printf("  Node: %s, bbox = (%.3f, %.3f, %.3f) - (%.3f, %.3f, %.3f)\n", node->getName().c_str(),
					bbox.lower[0], bbox.lower[1], bbox.lower[2],
					bbox.upper[0], bbox.upper[1], bbox.upper[2]);

				// set selected node
				activeWindow->currNode = node;
			}
			else {
				activeWindow->currNode = 0;
			}
		}

		auto camera = activeWindow->getCamera();

		activeWindow->setButtonType(button);
		activeWindow->setButtonAction(action);
		activeWindow->setButtonMode(mods);

		PMouseEvent mouseEvent;
		mouseEvent.ray = camera->castRayInWorldSpace((float)xpos, (float)ypos);
		mouseEvent.buttonType = (PButtonType)button;
		mouseEvent.actionType = (PActionType)action;
		mouseEvent.mods = (PModifierBits)activeWindow->getButtonMode();
		mouseEvent.camera = camera;
		mouseEvent.x = (float)xpos;
		mouseEvent.y = (float)ypos;

		auto activeScene = SceneGraphFactory::instance()->active();

		activeScene->onMouseEvent(mouseEvent);

		if (action == GLFW_PRESS)
		{
			// if(mOpenCameraRotate)
			camera->registerPoint((float)xpos, (float)ypos);
			activeWindow->setButtonState(GLFW_DOWN);

			activeWindow->imWindow()->mousePressEvent(mouseEvent);
		}
		else
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		if (action == GLFW_RELEASE)
		{
			activeWindow->setButtonState(GLFW_UP);

			activeWindow->imWindow()->mouseReleaseEvent(mouseEvent);
		}


		// update cursor position record
		if (action == GLFW_PRESS)
			activeWindow->setCursorPos(xpos, ypos);
		else
			activeWindow->setCursorPos(-1, -1);
	}

	void GlfwApp::cursorPosCallback(GLFWwindow* window, double x, double y)
	{
		GlfwApp* activeWindow = (GlfwApp*)glfwGetWindowUserPointer(window); // User Pointer
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
			activeWindow->getButtonType() == GLFW_MOUSE_BUTTON_RIGHT && 
			activeWindow->getButtonState() == GLFW_DOWN && 
			activeWindow->getButtonMode() == GLFW_MOD_ALT &&
			!activeWindow->mImWindow.cameraLocked()) 
		{
			camera->translateToPoint(x, y);
		}

		activeWindow->imWindow()->mouseMoveEvent(mouseEvent);
	}

	void GlfwApp::cursorEnterCallback(GLFWwindow* window, int entered)
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

	void GlfwApp::scrollCallback(GLFWwindow* window, double offsetX, double OffsetY)
	{
		GlfwApp* activeWindow = (GlfwApp*)glfwGetWindowUserPointer(window);
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

	void GlfwApp::keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		GlfwApp* activeWindow = (GlfwApp*)glfwGetWindowUserPointer(window);

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

	void GlfwApp::reshapeCallback(GLFWwindow* window, int w, int h)
	{
		GlfwApp* app = (GlfwApp*)glfwGetWindowUserPointer(window);
		app->setWindowSize(w, h);
	}

}