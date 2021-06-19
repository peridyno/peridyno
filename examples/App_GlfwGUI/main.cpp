#include "GlfwGUI/GlfwApp.h"

using namespace dyno;

int main(int, char**)
{
	GlfwApp window;

	window.setCameraType(CameraType::TrackBall);
	window.createWindow(1024, 768);

	window.mainLoop();
	return 0;
}
