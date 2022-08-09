#include <GlfwApp.h>

// GLFW
#include <GLFW/glfw3.h>


using namespace dyno;

int main(int, char**)
{

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GlfwApp window;
	window.createWindow(1024, 768);
	window.mainLoop();

	//printf("yys----------------------------------\n");
	//std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
	
	return 0;
}
