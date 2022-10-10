#include <QtApp.h>
using namespace dyno;

int main()
{
	QtApp window;
	window.createWindow(1366, 800);
	window.mainLoop();

	return 0;
}