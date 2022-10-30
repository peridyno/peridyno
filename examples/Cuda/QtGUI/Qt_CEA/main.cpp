#include <QtApp.h>
using namespace dyno;

int main()
{
	QtApp window;
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}