#include <QtApp.h>
using namespace dyno;

#include "ParticleSystem/initializeParticleSystem.h"
#include "initializeModeling.h"
#include "initializeInteraction.h"

/**
 * @brief This example demonstrate how to load plugin libraries in a static way
 */

int main()
{
	PaticleSystem::initStaticPlugin();
	Modeling::initStaticPlugin();
	Interaction::initStaticPlugin();

	QtApp window;
	window.createWindow(1366, 800);
	window.mainLoop();

	return 0;
}