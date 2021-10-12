#include <QtApp.h>
#include <GLRenderEngine.h>

using namespace dyno;

int main()
{
	GLRenderEngine* engine = new GLRenderEngine;

	QtApp window;
	window.setRenderEngine(engine);
	window.createWindow(1024, 768);

	window.mainLoop();

	delete engine;
	return 0;
}