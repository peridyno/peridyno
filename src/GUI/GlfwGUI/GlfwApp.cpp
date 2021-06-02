#include "GlfwApp.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include "Image_IO/image_io.h"
#include "Framework/SceneGraph.h"
#include "Framework/Log.h"

#include "../RenderEngine/RenderEngine.h"
#include "../RenderEngine/RenderParams.h"

namespace dyno 
{
	static void glfw_error_callback(int error, const char* description)
	{
		fprintf(stderr, "Glfw Error %d: %s\n", error, description);
	}

	GlfwApp::GlfwApp(int argc /*= 0*/, char **argv /*= NULL*/)
	{

	}

	GlfwApp::GlfwApp(int width, int height)
	{
		this->createWindow(width, height);
	}

	GlfwApp::~GlfwApp()
	{
		// Cleanup
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		//
		delete mRenderEngine;
		delete mRenderParams;

		glfwDestroyWindow(mWindow);
		glfwTerminate();

	}

	void GlfwApp::createWindow(int width, int height)
	{
		mWidth = width;
		mHeight = height;

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
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
		//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
		//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

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

		initOpenGL();

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

		// Setup Platform/Renderer backends
		ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
		ImGui_ImplOpenGL3_Init(glsl_version);

		mCamera.registerPoint(0.5f, 0.5f);
		mCamera.translateToPoint(0, 0);

		mCamera.zoom(3.0f);
		mCamera.setGL(0.01f, 3.0f, (float)getWidth(), (float)getHeight());

		// Jian: initialize rendering engine
		mRenderEngine = new RenderEngine();
		mRenderParams = new RenderParams();

		mRenderEngine->initialize();

		// set the viewport
		mRenderParams->viewport.x = 0;
		mRenderParams->viewport.y = 0;
		mRenderParams->viewport.w = width;
		mRenderParams->viewport.h = height;
	}

	void GlfwApp::mainLoop()
	{
		SceneGraph::getInstance().initialize();

		mRenderEngine->setSceneGraph(&SceneGraph::getInstance());

		bool show_demo_window = true;

		// Main loop
		while (!glfwWindowShouldClose(mWindow))
		{
			glfwPollEvents();

			if (mAnimationToggle)
				SceneGraph::getInstance().takeOneFrame();

			int width, height;
			glfwGetFramebufferSize(mWindow, &width, &height);
			const float ratio = width / (float)height;

			glViewport(0, 0, width, height);
			glClearColor(mClearColor.x * mClearColor.w, mClearColor.y * mClearColor.w, mClearColor.z * mClearColor.w, mClearColor.w);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			
			drawScene();

			//// Start the Dear ImGui frame
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
			{
				static float f = 0.0f;
				static int counter = 0;

				ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

				ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
				ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state

				ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
				ImGui::ColorEdit3("clear color", (float*)&mClearColor); // Edit 3 floats representing a color

				if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
					counter++;
				ImGui::SameLine();
				ImGui::Text("counter = %d", counter);

				ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
				ImGui::End();
			}

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

	int GlfwApp::getWidth() const
	{
		return mWidth;
	}

	int GlfwApp::getHeight() const
	{
		return mHeight;
	}

	void GlfwApp::setWidth(int width)
	{
		mWidth = width;
	}

	void GlfwApp::setHeight(int height)
	{
		mHeight = height;
	}


	bool GlfwApp::saveScreen(const std::string &file_name) const
	{
		int width = this->getWidth(), height = this->getHeight();
		unsigned char *data = new unsigned char[width*height * 3];  //RGB
		assert(data);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, (void*)data);
		Image image(width, height, Image::RGB, data);
		image.flipVertically();
		bool status = ImageIO::save(file_name, &image);
		delete[] data;
		return status;
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


	void GlfwApp::toggleAnimation()
	{
		mAnimationToggle = !mAnimationToggle;
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

	void GlfwApp::initOpenGL()
	{
		glShadeModel(GL_SMOOTH);
		glClearDepth(1.0);														// specify the clear value for the depth buffer
		glEnable(GL_DEPTH_TEST);
	}

	void GlfwApp::drawScene(void)
	{
		if (mUseNewRenderEngine)
		{
			// simply dump transform matrices...
			glGetFloatv(GL_PROJECTION_MATRIX, &mRenderParams->proj[0][0]);
			glGetFloatv(GL_MODELVIEW_MATRIX, &mRenderParams->view[0][0]);
			
			// set the viewport
			mRenderParams->viewport.x = 0;
			mRenderParams->viewport.y = 0;
			mRenderParams->viewport.w = mCamera.mViewportWidth;
			mRenderParams->viewport.h = mCamera.mViewportHeight;
			
			mRenderEngine->render(*mRenderParams);
		}
		else
		{
			glUseProgram(0);
			drawBackground();

			SceneGraph::getInstance().draw();
		}
	}

	void GlfwApp::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
	{
		GlfwApp* activeWindow = (GlfwApp*)glfwGetWindowUserPointer(window);
		auto camera = activeWindow->activeCamera();

		activeWindow->setButtonType(button);
		activeWindow->setButtonAction(action);
		activeWindow->setButtonMode(mods);

		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		if (action == GLFW_PRESS)
		{
			camera->registerPoint(float(xpos) / float(activeWindow->getWidth()) - 0.5f, float(activeWindow->getHeight() - float(ypos)) / float(activeWindow->getHeight()) - 0.5f);
			activeWindow->setButtonState(GLFW_DOWN);
		}
		else
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		if (action == GLFW_RELEASE)
		{
			activeWindow->setButtonState(GLFW_UP);
		}

		if (action != GLFW_PRESS)
			return;
	}

	void GlfwApp::cursorPosCallback(GLFWwindow* window, double x, double y)
	{
		GlfwApp* activeWindow = (GlfwApp*)glfwGetWindowUserPointer(window);

		auto camera = activeWindow->activeCamera();

		if (activeWindow->getButtonType() == GLFW_MOUSE_BUTTON_LEFT && activeWindow->getButtonState() == GLFW_DOWN) {
			camera->rotateToPoint(float(x) / float(activeWindow->getWidth()) - 0.5f, float(activeWindow->getHeight() - y) / float(activeWindow->getHeight()) - 0.5f);
		}
		else if (activeWindow->getButtonType() == GLFW_MOUSE_BUTTON_RIGHT && activeWindow->getButtonState() == GLFW_DOWN) {
			camera->translateToPoint(float(x) / float(activeWindow->getWidth()) - 0.5f, float(activeWindow->getHeight() - y) / float(activeWindow->getHeight()) - 0.5f);
		}
		else if (activeWindow->getButtonType() == GLFW_MOUSE_BUTTON_MIDDLE) {
			camera->translateLightToPoint(float(x) / float(activeWindow->getWidth()) - 0.5f, float(activeWindow->getHeight() - y) / float(activeWindow->getHeight()) - 0.5f);
		}
		camera->setGL(0.01f, 10.0f, (float)activeWindow->getWidth(), (float)activeWindow->getHeight());
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
		auto camera = activeWindow->activeCamera();

		camera->zoom(-OffsetY);
		camera->setGL(0.01f, 10.0f, (float)activeWindow->getWidth(), (float)activeWindow->getHeight());
	}

	void GlfwApp::keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		GlfwApp* activeWindow = (GlfwApp*)glfwGetWindowUserPointer(window);

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
		default:
			break;
		}
	}

	void GlfwApp::reshapeCallback(GLFWwindow* window, int w, int h)
	{
		GlfwApp* activeWindow = (GlfwApp*)glfwGetWindowUserPointer(window);

		glfwGetFramebufferSize(window, &activeWindow->mWidth, &activeWindow->mHeight);

		activeWindow->activeCamera()->setGL(0.01f, 10.0f, (float)w, (float)h);
		activeWindow->setWidth(w);
		activeWindow->setHeight(h);

		glViewport(0, 0, w, h);
	}

	void GlfwApp::drawBackground()
	{
		int xmin = -mPlaneSize;
		int xmax = mPlaneSize;
		int zmin = -mPlaneSize;
		int zmax = mPlaneSize;

		float s = 1.0f;
		int nSub = 10;
		float sub_s = s / nSub;

		glPushMatrix();

		float ep = 0.0001f;
		glPushMatrix();
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

		//Draw background grid
		glLineWidth(2.0f);
		glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
		glBegin(GL_LINES);
		for (int i = xmin; i <= xmax; i++)
		{
			glVertex3f(i*s, 0, zmin*s);
			glVertex3f(i*s, 0, zmax*s);
		}
		for (int i = zmin; i <= zmax; i++)
		{
			glVertex3f(xmin*s, 0, i*s);
			glVertex3f(xmax*s, 0, i*s);
		}

		glEnd();

		glLineWidth(1.0f);
		glLineStipple(1, 0x5555);
		glEnable(GL_LINE_STIPPLE);
		glColor4f(0.55f, 0.55f, 0.55f, 1.0f);
		glBegin(GL_LINES);
		for (int i = xmin; i < xmax; i++)
		{
			for (int j = 1; j < nSub; j++)
			{
				glVertex3f(i*s + j * sub_s, 0, zmin*s);
				glVertex3f(i*s + j * sub_s, 0, zmax*s);
			}
		}
		for (int i = zmin; i < zmax; i++)
		{
			for (int j = 1; j < nSub; j++)
			{
				glVertex3f(xmin*s, 0, i*s + j * sub_s);
				glVertex3f(xmax*s, 0, i*s + j * sub_s);
			}
		}
		glEnd();
		glDisable(GL_LINE_STIPPLE);

		glPopMatrix();

//		drawAxis();
	}

	void GlfwApp::drawAxis()
	{
		GLfloat mv[16];
		GLfloat proj[16];
		glGetFloatv(GL_PROJECTION_MATRIX, proj);
		glGetFloatv(GL_MODELVIEW_MATRIX, mv);
		mv[12] = mv[13] = mv[14] = 0.0;

		glPushAttrib(GL_ALL_ATTRIB_BITS);

		glMatrixMode(GL_PROJECTION);

		glPushMatrix();
		glLoadIdentity();
		glOrtho(0.0, 0.0, 1.0, 1.0, -1.0, 1.0);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadMatrixf(mv);

		//Draw axes
		glViewport(20, 10, 90, 80);
		glColor3ub(255, 255, 255);
		glLineWidth(1.0f);
		const float len = 0.9f;
		GLfloat origin[3] = { 0.0f, 0.0f, 0.0f };
		glBegin(GL_LINES);
		glColor3f(1, 0, 0);
		glVertex3f(origin[0], origin[1], origin[2]);
		glVertex3f(origin[0] + len, origin[1], origin[2]);
		glColor3f(0, 1, 0);
		glVertex3f(origin[0], origin[1], origin[2]);
		glVertex3f(origin[0], origin[1] + len, origin[2]);
		glColor3f(0, 0, 1);
		glVertex3f(origin[0], origin[1], origin[2]);
		glVertex3f(origin[0], origin[1], origin[2] + len);
		glEnd();

// 		// Draw labels
// 		glColor3f(1, 0, 0);
// 		glRasterPos3f(origin[0] + len, origin[1], origin[2]);
// 		glfwBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'x');
// 		glColor3f(0, 1, 0);
// 		glRasterPos3f(origin[0], origin[1] + len, origin[2]);
// 		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'y');
// 		glColor3f(0, 0, 1);
// 		glRasterPos3f(origin[0], origin[1], origin[2] + len);
// 		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'z');

		glPopAttrib();

		// Restore viewport, projection and model-view matrices
		glViewport(0, 0, mWidth, mHeight);
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
	}
}