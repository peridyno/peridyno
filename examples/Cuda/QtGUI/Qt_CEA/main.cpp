#include <QtApp.h>
using namespace dyno;

int main()
{
	QtApp app;
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}