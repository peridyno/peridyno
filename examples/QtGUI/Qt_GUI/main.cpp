#include <QtApp.h>
using namespace dyno;

#include "ParticleSystem/initializeParticleSystem.h"

int main()
{
	PaticleSystem::initStaticPlugin();

	QtApp window;
	window.createWindow(1366, 800);
	window.mainLoop();

	return 0;
}