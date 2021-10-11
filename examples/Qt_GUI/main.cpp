//#include <QtApp.h>
//#include <GLRenderEngine.h>
#include <QtApp.h>

#include <SceneGraph.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/ParticleEmitterSquare.h>

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>

#include <ImColorbar.h>
using namespace dyno;

int main()
{
	GLRenderEngine* engine = new GLRenderEngine;

	QtApp window;
	window.setRenderEngine(engine);
	window.createWindow(1024, 768);

	window.mainLoop();

	delete engine;
	return 0;
}