#include <GlfwApp.h>

#include <SceneGraph.h>

using namespace px;

int main(int, char**)
{
	dyno::GlfwApp window;
	window.createWindow(1024, 768);
	window.mainLoop();
	return 0;
}
