#include "GlfwApp.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include "Image_IO/image_io.h"
#include "SceneGraph.h"
#include "Log.h"

#include "camera/OrbitCamera.h"
#include "camera/TrackballCamera.h"

#include "../RenderEngine/RenderEngine.h"
#include "../RenderEngine/RenderTarget.h"
#include "../RenderEngine/RenderParams.h"

namespace dyno 
{
	static void glfw_error_callback(int error, const char* description)
	{
		fprintf(stderr, "Glfw Error %d: %s\n", error, description);
	}

	GlfwApp::GlfwApp(int argc /*= 0*/, char **argv /*= NULL*/)
	{
		setupCamera();
	}

	GlfwApp::GlfwApp(int width, int height)
	{
		setupCamera();
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
		delete mRenderTarget;
		delete mRenderParams;

		glfwDestroyWindow(mWindow);
		glfwTerminate();

	}

	void GlfwApp::createWindow(int width, int height)
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
		initializeStyle();
		loadIcon();

		// Setup Platform/Renderer backends
		ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
		ImGui_ImplOpenGL3_Init(glsl_version);

		mCamera->setWidth(width);
		mCamera->setHeight(height);
		mCamera->registerPoint(0.5f, 0.5f);
		mCamera->translateToPoint(0, 0);

		mCamera->zoom(3.0f);
		mCamera->setClipNear(0.01f);
		mCamera->setClipFar(10.0f);

		// Jian: initialize rendering engine
		mRenderEngine = new RenderEngine();
		mRenderTarget = new RenderTarget();
		mRenderParams = new RenderParams();

		mRenderEngine->initialize();
		mRenderTarget->initialize();

		mRenderTarget->resize(width, height);
		// set the viewport
		mRenderParams->viewport.x = 0;
		mRenderParams->viewport.y = 0;
		mRenderParams->viewport.w = width;
		mRenderParams->viewport.h = height;
	}

	void GlfwApp::setupCamera()
	{
		switch (mCameraType)
		{
		case dyno::Orbit:
			mCamera = std::make_shared<OrbitCamera>();
			break;
		case dyno::TrackBall:
			mCamera = std::make_shared<TrackballCamera>();
			break;
		default:
			break;
		}
	}

	void GlfwApp::loadIcon(){
		pics.emplace_back(std::make_shared<Picture>("../../data/icon/map.png"));
		pics.emplace_back(std::make_shared<Picture>("../../data/icon/box.png"));
		pics.emplace_back(std::make_shared<Picture>("../../data/icon/arrow-090-medium.png"));
		pics.emplace_back(std::make_shared<Picture>("../../data/icon/lock.png"));
		// pics.emplace_back(std::make_shared<Picture>("../../../data/icon/map.png"));
		// pics.emplace_back(std::make_shared<Picture>("../../../data/icon/box.png"));
	}

	void GlfwApp::initializeStyle()
	{
		ImGuiStyle& style = ImGui::GetStyle();
		style.WindowRounding = 6.0f;
		style.ChildRounding = 6.0f;
		style.FrameRounding = 6.0f;
		style.PopupRounding = 6.0f;
	}
	void GlfwApp::toggleButton(ImTextureID texId, const char* label, bool *v)
	{
		if (*v == true)
		{

			ImGui::PushID(label);
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(230/255.0, 179/255.0, 0/255.0, 105/255.0));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(230/255.0, 179/255.0, 0/255.0, 255/255.0));
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(255/255.0, 153/255.0, 0/255.0, 255/255.0));
			ImGui::ImageButtonWithText(texId, label);
			if (ImGui::IsItemClicked(0))
			{
				*v = !*v;
			}
			ImGui::PopStyleColor(3);
			ImGui::PopID();
		}
		else
		{
			if (ImGui::ImageButtonWithText(texId ,label))
				*v = true;
		}
	}
	void GlfwApp::toggleButton(const char* label, bool *v)
	{
		if (*v == true)
		{

			ImGui::PushID(label);
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(230/255.0, 179/255.0, 0/255.0, 105/255.0));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(230/255.0, 179/255.0, 0/255.0, 255/255.0));
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(255/255.0, 153/255.0, 0/255.0, 255/255.0));
			ImGui::Button(label);
			if (ImGui::IsItemClicked(0))
			{
				*v = !*v;
			}
			ImGui::PopStyleColor(3);
			ImGui::PopID();
		}
		else
		{
			if (ImGui::Button(label))
				*v = true;
		}
	}

	void GlfwApp::sampleButton(const char* label, bool *v)
	{
		float padding = 10.0f;
		float bounding = 1.0f;
		ImVec2 p = ImGui::GetCursorScreenPos();
		ImDrawList* draw_list = ImGui::GetWindowDrawList();
		const ImVec2 label_size = ImGui::CalcTextSize(label);
		const ImVec2 button_size = ImVec2(label_size.x + padding * 2, label_size.y + padding * 2);
		const ImVec2 bound_size =  ImVec2(button_size.x + bounding * 2, button_size.y + bounding * 2);
		ImVec2 p_button = ImVec2(p.x + bounding, p.y + bounding);
		ImVec2 p_label = ImVec2(p_button.x + padding, p_button.y + padding);

		float radius = bound_size.y * 0.30f;

		// 透明的按钮
		if (ImGui::InvisibleButton(label, bound_size))
			*v = !*v;
		ImVec4 col_bf4;
		ImGuiStyle& style = ImGui::GetStyle();

		// 颜色自定义
		if (ImGui::IsItemActivated()) col_bf4 = *v ? style.Colors[40] : style.Colors[23];
		else if (ImGui::IsItemHovered()) col_bf4 =  *v ? style.Colors[42] : style.Colors[24];
		else col_bf4 = *v ? style.Colors[41] : style.Colors[22];

		ImU32 col_bg = IM_COL32(255 * col_bf4.x, 255 * col_bf4.y, 255 * col_bf4.z, 255 * col_bf4.w);
		ImU32 col_text = IM_COL32(255, 255, 255, 255);
		ImU32 col_bound = IM_COL32(0,0,0,255);
		
		// 绘制矩形形状
		draw_list->AddRect(p, ImVec2(p.x + bound_size.x, p.y + bound_size.y), col_bound , radius);
		draw_list->AddRectFilled(p_button, ImVec2(p_button.x + button_size.x, p_button.y + button_size.y), col_bg, radius);
		draw_list->AddText(p_label, col_text, label);
	}

	void GlfwApp::beginTitle(const char* label){
		ImGui::PushID(label);
	}

	void GlfwApp::endTitle(){
		ImGui::PopID();
	}

	void GlfwApp::mainLoop()
	{
		
		SceneGraph::getInstance().initialize();

		float iBgGray[2] = { 0.2f, 0.8f };
		bool mLock = false;
		RenderParams::Light iLight;
		int width = 1024, height = 768;
		static float values[90] = {};
        static int values_offset = 0;
        static double refresh_time = 0.0;

		// Main loop
		while (!glfwWindowShouldClose(mWindow))
		{
			
			glfwPollEvents();

			if (mAnimationToggle)
				SceneGraph::getInstance().takeOneFrame();
				

			// Start the Dear ImGui frame
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
			{
				static float f = 0.0f;
				static int counter = 0;

				{// Top Left widget
					ImGui::SetNextWindowPos(ImVec2(0,0));
					ImGui::Begin("Top Left widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
					
					if(ImGui::Button("Lighting")){
						ImGui::OpenPopup("LightingMenu");
					}

					if(ImGui::BeginPopup("LightingMenu")){
						ImGui::SliderFloat2("BG color", iBgGray, 0.0f, 1.0f, "%.3f", 0);
						mRenderParams->bgColor0 = glm::vec3(iBgGray[0]);
						mRenderParams->bgColor1 = glm::vec3(iBgGray[1]);
						
						ImGui::Text("Ambient Light");

						beginTitle("Ambient Light Scale");
						ImGui::SliderFloat("", &iLight.ambientScale, 0.0f, 10.0f, "%.3f", 0); 
						endTitle();
						ImGui::SameLine();
						ImGui::ColorEdit3("Ambient Light Color", (float*)&iLight.ambientColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float | ImGuiColorEditFlags_NoDragDrop | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoLabel) ;

						ImGui::Text("Main Light");
						beginTitle("Main Light Scale");
						ImGui::SliderFloat("", &iLight.mainLightScale, 0.0f, 10.0f, "%.3f", 0); 
						endTitle();
						ImGui::SameLine();
						ImGui::ColorEdit3("Main Light Color", (float*)&iLight.mainLightColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float | ImGuiColorEditFlags_NoDragDrop | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoLabel);
						mRenderParams->light = iLight;

						ImGui::EndPopup();
					}
					

					// Camera Select
					static int camera_current = 0;
					const char* camera_name[] = {"Orbit", "TrackBall"};
					static ImGuiComboFlags flags = ImGuiComboFlags_NoArrowButton;
					// ImGui::Combo("Camera", &camera_current, camera_name, IM_ARRAYSIZE(camera_name));
					ImGui::SetNextItemWidth(100);

					beginTitle("Camera");
					if (ImGui::BeginCombo("", camera_name[camera_current], flags))
					{
						for (int n = 0; n < IM_ARRAYSIZE(camera_name); n++)
						{
							const bool is_selected = (camera_current == n);
							if (ImGui::Selectable(camera_name[n], is_selected))
								camera_current = n;
							// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
							if (is_selected)
								ImGui::SetItemDefaultFocus();
						}
						ImGui::EndCombo();
					}			
					endTitle();

					if(CameraType(camera_current) != mCameraType){
						// FIXME: GL error
						// setCameraType(CameraType(camera_current));
					}
					ImGui::End();
				}

				{// Top Right widget
					
					ImGui::Begin("Top Right widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
					toggleButton(pics[3]->GetTexture(),"Lock", &(mLock));
					ImGui::SameLine();
					toggleButton(pics[0]->GetTexture(),"Ground", &(mRenderParams->showGround));
					ImGui::SameLine();
					toggleButton(pics[1]->GetTexture(),"Bounds",&(mRenderParams->showSceneBounds));
					ImGui::SameLine();
					toggleButton(pics[2]->GetTexture(),"Axis Helper", &(mRenderParams->showAxisHelper));
					ImGui::SetWindowPos(ImVec2(width - ImGui::GetWindowSize().x, 0));
						
					if (refresh_time == 0.0) refresh_time = ImGui::GetTime();
					while (refresh_time < ImGui::GetTime()) // Create data at fixed 60 Hz rate for the demo
					{
						static float phase = 0.0f;
						values[values_offset] = cosf(phase);
						values_offset = (values_offset + 1) % IM_ARRAYSIZE(values);
						phase += 0.10f * values_offset;
						refresh_time += 1.0f / 60.0f;
					}
					char overlay[32];
					ImGui::PlotLines("Lines",  values, IM_ARRAYSIZE(values), values_offset, NULL, -1.0f, 1.0f, ImVec2(0, 80.0f));

					ImGui::End();
				}

				{// Bottom Right widget
					ImGui::Begin("Bottom Left widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
					ImGui::Text(" %.1f FPS", ImGui::GetIO().Framerate);
					ImGui::SetWindowPos(ImVec2(width - ImGui::GetWindowSize().x, height - ImGui::GetWindowSize().y));
					ImGui::End();
				}

				// Mouse Foucus on Any Imgui Windows || Lock
				mOpenCameraRotate = !(ImGui::IsWindowFocused(ImGuiFocusedFlags_::ImGuiFocusedFlags_AnyWindow) || mLock);
			}

			ImGui::Render();

			
			glfwGetFramebufferSize(mWindow, &width, &height);
			const float ratio = width / (float)height;

			glViewport(0, 0, width, height);
			glClearColor(mClearColor.x * mClearColor.w, mClearColor.y * mClearColor.w, mClearColor.z * mClearColor.w, mClearColor.w);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			
			drawScene();

			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

			glfwSwapBuffers(mWindow);
		}
	}

	void GlfwApp::setCameraType(CameraType type)
	{
		mCameraType = type;
		setupCamera();
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


	void GlfwApp::setWindowSize(int width, int height)
	{
		mCamera->setWidth(width);
		mCamera->setHeight(height);
	}

	bool GlfwApp::saveScreen(const std::string &file_name) const
	{
		int width;
		int height;
		glfwGetFramebufferSize(mWindow, &width, &height);

		unsigned char *data = new unsigned char[width * height * 3];  //RGB
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

	bool GlfwApp::getCameraRotateFlag() {
		return mOpenCameraRotate;
	}

	int GlfwApp::getWidth()
	{
		return activeCamera()->viewportWidth();
	}

	int GlfwApp::getHeight()
	{
		return activeCamera()->viewportHeight();
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
		// preserve current framebuffer
		GLint fbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);

		mRenderParams->proj = mCamera->getProjMat();
		mRenderParams->view = mCamera->getViewMat();
						
		// set the viewport
		mRenderParams->viewport.x = 0;
		mRenderParams->viewport.y = 0;
		mRenderParams->viewport.w = mCamera->viewportWidth();
		mRenderParams->viewport.h = mCamera->viewportHeight();

		mRenderTarget->resize(mCamera->viewportWidth(), mCamera->viewportHeight());
		
		mRenderEngine->draw(&SceneGraph::getInstance(), mRenderTarget, *mRenderParams);

		// write back to the framebuffer
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
		mRenderTarget->blit(0);
		
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
			// if(mOpenCameraRotate)
			camera->registerPoint(xpos, ypos);
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
		GlfwApp* activeWindow = (GlfwApp*)glfwGetWindowUserPointer(window); // User Pointer

		auto camera = activeWindow->activeCamera();

		if (activeWindow->getButtonType() == GLFW_MOUSE_BUTTON_LEFT && activeWindow->getButtonState() == GLFW_DOWN && mOpenCameraRotate) {
			camera->rotateToPoint(x, y);
		}
		else if (activeWindow->getButtonType() == GLFW_MOUSE_BUTTON_RIGHT && activeWindow->getButtonState() == GLFW_DOWN && mOpenCameraRotate) {
			camera->translateToPoint(x, y);
		}

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
		if(mOpenCameraRotate)camera->zoom(-OffsetY);
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
		activeWindow->setWindowSize(w, h);
	}

}