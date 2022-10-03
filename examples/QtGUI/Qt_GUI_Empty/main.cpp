#include <QtApp.h>
using namespace dyno;

int main()
{
	QtApp window;
	//Will not load the plugins
	window.createWindow(1366, 800, false);
	window.mainLoop();

	return 0;
}