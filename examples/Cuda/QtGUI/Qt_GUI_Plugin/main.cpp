#include <QtApp.h>
using namespace dyno;

#include "ParticleSystem/initializeParticleSystem.h"
#include "SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h"
#include "initializeModeling.h"
#include "initializeInteraction.h"

/**
 * @brief This example demonstrate how to load plugin libraries in a static way
 */

int main()
{
	PaticleSystem::initStaticPlugin();
	SemiAnalyticalScheme::initStaticPlugin();
	Modeling::initStaticPlugin();
	Interaction::initStaticPlugin();

	QtApp app;
	app.initialize(1366, 800);
	app.mainLoop();

	return 0;
}