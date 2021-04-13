#include "GlfwApp.h"

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

		glfwDestroyWindow(window);
		glfwTerminate();

	}

	void GlfwApp::createWindow(int width, int height)
	{
		mWidth = width;
		mHeight = height;

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
		window = glfwCreateWindow(width, height, "Dear ImGui GLFW+OpenGL3 example", NULL, NULL);
		if (window == NULL)
			return;

		initCallbacks();

		glfwMakeContextCurrent(window);
		glfwSwapInterval(1); // Enable vsync

		glfwSetWindowUserPointer(window, this);

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
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init(glsl_version);
	}

	void GlfwApp::mainLoop()
	{
		// Our state
		bool show_demo_window = true;
		bool show_another_window = false;

		// Main loop
		while (!glfwWindowShouldClose(window))
		{
			// Poll and handle events (inputs, window resize, etc.)
			// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
			// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
			// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
			// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
			glfwPollEvents();

			// Start the Dear ImGui frame
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
			if (show_demo_window)
				ImGui::ShowDemoWindow(&show_demo_window);

			// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
			{
				static float f = 0.0f;
				static int counter = 0;

				ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

				ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
				ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
				ImGui::Checkbox("Another Window", &show_another_window);

				ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
				ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

				if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
					counter++;
				ImGui::SameLine();
				ImGui::Text("counter = %d", counter);

				ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
				ImGui::End();
			}

			// 3. Show another simple window.
			if (show_another_window)
			{
				ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
				ImGui::Text("Hello from another window!");
				if (ImGui::Button("Close Me"))
					show_another_window = false;
				ImGui::End();
			}

			// Rendering
			ImGui::Render();
			
			drawScene();

			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

			glfwSwapBuffers(window);
		}
	}

	void GlfwApp::initCallbacks()
	{
		mMouseButtonCallback = GlfwApp::mouseButtonCallback;
		mKeyboardCallback = GlfwApp::keyboardCallback;

		glfwSetMouseButtonCallback(window, mMouseButtonCallback);
		glfwSetKeyCallback(window, mKeyboardCallback);
	}

#define TORUS_MAJOR     1.5
#define TORUS_MINOR     0.5
#define TORUS_MAJOR_RES 32
#define TORUS_MINOR_RES 32


	static void drawTorus(void)
	{
		static GLuint torus_list = 0;
		int    i, j, k;
		double s, t, x, y, z, nx, ny, nz, scale, twopi;

		if (!torus_list)
		{
			// Start recording displaylist
			torus_list = glGenLists(1);
			glNewList(torus_list, GL_COMPILE_AND_EXECUTE);

			// Draw torus
			twopi = 2.0 * M_PI;
			for (i = 0; i < TORUS_MINOR_RES; i++)
			{
				glBegin(GL_QUAD_STRIP);
				for (j = 0; j <= TORUS_MAJOR_RES; j++)
				{
					for (k = 1; k >= 0; k--)
					{
						s = (i + k) % TORUS_MINOR_RES + 0.5;
						t = j % TORUS_MAJOR_RES;

						// Calculate point on surface
						x = (TORUS_MAJOR + TORUS_MINOR * cos(s * twopi / TORUS_MINOR_RES)) * cos(t * twopi / TORUS_MAJOR_RES);
						y = TORUS_MINOR * sin(s * twopi / TORUS_MINOR_RES);
						z = (TORUS_MAJOR + TORUS_MINOR * cos(s * twopi / TORUS_MINOR_RES)) * sin(t * twopi / TORUS_MAJOR_RES);

						// Calculate surface normal
						nx = x - TORUS_MAJOR * cos(t * twopi / TORUS_MAJOR_RES);
						ny = y;
						nz = z - TORUS_MAJOR * sin(t * twopi / TORUS_MAJOR_RES);
						scale = 1.0 / sqrt(nx*nx + ny * ny + nz * nz);
						nx *= scale;
						ny *= scale;
						nz *= scale;

						glNormal3f((float)nx, (float)ny, (float)nz);
						glVertex3f((float)x, (float)y, (float)z);
					}
				}

				glEnd();
			}

			// Stop recording displaylist
			glEndList();
		}
		else
		{
			// Playback displaylist
			glCallList(torus_list);
		}
	}

	void GlfwApp::drawScene(void)
	{
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);

		glPushMatrix();

		// Move back
		glTranslatef(0.0, 0.0, -zoom);
		// Rotate the view
		glRotatef(beta, 1.0, 0.0, 0.0);
		glRotatef(alpha, 0.0, 0.0, 1.0);

		//drawTorus();
		drawBackground();

		glPopMatrix();
	}

	void GlfwApp::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
	{
		GlfwApp* activeWindow = (GlfwApp*)glfwGetWindowUserPointer(window);

		if (action != GLFW_PRESS)
			return;
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
			break;
		case GLFW_KEY_LEFT:
			activeWindow->alpha += 5;
			break;
		case GLFW_KEY_RIGHT:
			activeWindow->alpha -= 5;
			break;
		case GLFW_KEY_UP:
			activeWindow->beta -= 5;
			break;
		case GLFW_KEY_DOWN:
			activeWindow->beta += 5;
			break;
		case GLFW_KEY_PAGE_UP:
			activeWindow->zoom -= 0.25f;
			if (activeWindow->zoom < 0.f)
				activeWindow->zoom = 0.f;
			break;
		case GLFW_KEY_PAGE_DOWN:
			activeWindow->zoom += 0.25f;
			break;
		default:
			break;
		}
	}


	void GlfwApp::drawBackground()
	{
		int xmin = -10;
		int xmax = 10;
		int zmin = -10;
		int zmax = 10;

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
		for (int i = xmin; i <= xmax; i++)
		{
			for (int j = 1; j < nSub; j++)
			{
				glVertex3f(i*s + j * sub_s, 0, zmin*s);
				glVertex3f(i*s + j * sub_s, 0, zmax*s);
			}
		}
		for (int i = zmin; i <= zmax; i++)
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