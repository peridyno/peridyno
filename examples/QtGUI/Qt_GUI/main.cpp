#include <QtApp.h>
#include <Plugin/PluginManager.h>
using namespace dyno;

int main()
{
	/*PluginManager::instance()->loadPlugin("dynoIO-0.6.1d");*/
	PluginManager::instance()->loadPlugin("dynoInteraction-0.6.1d");
	PluginManager::instance()->loadPlugin("dynoModeling-0.6.1d");

	QtApp window;
	window.createWindow(1366, 800);
	window.mainLoop();

	return 0;
}