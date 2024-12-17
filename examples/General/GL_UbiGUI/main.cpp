#include <UbiApp.h>

using namespace dyno;

int main(int, char**)
{
	UbiApp app(GUIType::GUI_GLFW);
	app.initialize(1024, 768);
	app.mainLoop();
	return 0;
}
