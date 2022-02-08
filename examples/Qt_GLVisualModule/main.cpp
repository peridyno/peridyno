#include <QtApp.h>
#include <GlfwApp.h>

#include <SceneGraph.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/ParticleEmitterSquare.h>

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>

#include <ImColorbar.h>

#include "GLPointVisualNode.h"

using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vec3f(1.5, 1, 1.5));
	scene.setLowerBound(Vec3f(-0.5, 0, -0.5));

	auto fluid = scene.addNode(std::make_shared<ParticleFluid<DataType3f>>());
	fluid->loadParticles(Vec3f(0.5, 0.2, 0.4), Vec3f(0.7, 1.5, 0.6), 0.005);

	auto visualizer = scene.addNode(std::make_shared<GLPointVisualNode<DataType3f>>());
	visualizer->setParticles(fluid);

	auto boundary = scene.addNode(std::make_shared<StaticBoundary<DataType3f>>());
	boundary->loadCube(Vec3f(-0.5, 0, -0.5), Vec3f(1.5, 2, 1.5), 0.02, true);
	boundary->loadSDF("../../data/bowl/bowl.sdf", false);
	boundary->addParticleSystem(fluid);

	auto outTop = fluid->currentTopology()->promoteToOuput();
	outTop->connect(visualizer->inPointSetIn());
	outTop->disconnect(visualizer->inPointSetIn());
}

int main()
{
	RenderEngine* engine = new GLRenderEngine;

	CreateScene();

	QtApp window;
	window.setRenderEngine(engine);
	window.createWindow(1024, 768);
	window.mainLoop();

	delete engine;

	return 0;
}