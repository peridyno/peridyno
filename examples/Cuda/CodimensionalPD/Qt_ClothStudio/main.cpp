#include <QtApp.h>

#include <Peridynamics/initializePeridynamics.h>
#include <initializeModeling.h>
#include "Volume/initializeVolume.h"
#include "Multiphysics/initializeMultiphysics.h"

using namespace std;
using namespace dyno;

int main()
{
	Modeling::initStaticPlugin();
	Peridynamics::initStaticPlugin();
	Volume::initStaticPlugin();
	Multiphysics::initStaticPlugin();

	QtApp app;
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}