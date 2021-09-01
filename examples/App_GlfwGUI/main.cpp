#include <GlfwApp.h>
#include <GLRenderEngine.h>

using namespace dyno;

int main(int, char**)
{
	GLRenderEngine* engine = new GLRenderEngine;

	GlfwApp window;
	window.setRenderEngine(engine);
	window.createWindow(1024, 768);
	window.mainLoop();

	delete engine;

	return 0;
}
