#include <GlfwApp.h>

using namespace dyno;

int main(int, char**)
{
	GlfwApp app;
	app.initialize(1024, 768);
	app.mainLoop();
	return 0;
}
