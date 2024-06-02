#include <QtApp.h>
using namespace dyno;

/**
 * This example demonstrates how to configure a camera.
 */
int main(int, char**)
{
	QtApp app;
	app.initialize(1024, 768);

	//Set the eye position for the camera
	app.renderWindow()->getCamera()->setEyePos(Vec3f(0.61f, 1.09f, 1.74f));

	//Set the target position for the camera
	app.renderWindow()->getCamera()->setTargetPos(Vec3f(-0.12, 0.01f, -0.06f));

	//Set the distance unit for the camera, the fault unit is meter
	app.renderWindow()->getCamera()->setUnitScale(3.0f);

	app.mainLoop();

	return 0;
}
