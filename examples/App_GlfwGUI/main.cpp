#include "GlfwGUI/GlfwApp.h"

using namespace dyno;
bool GlfwApp::mOpenCameraRotate = true;

int main(int, char**)
{
	GlfwApp window;

	window.setCameraType(CameraType::TrackBall);
	window.createWindow(1024, 768);

	window.mainLoop();
	return 0;
}
