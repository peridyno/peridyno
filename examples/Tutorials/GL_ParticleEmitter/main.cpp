#include <GlfwApp.h>

#include <SceneGraph.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/SquareEmitter.h>

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	//Create a particle emitter
	auto emitter = scn->addNode(std::make_shared<SquareEmitter<DataType3f>>());
	emitter->varLocation()->setValue(Vec3f(0.5f));

	//Create a particle-based fluid solver
	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	fluid->loadParticles(Vec3f(0.0f), Vec3f(0.2f), 0.005f);
	emitter->connect(fluid->importParticleEmitters());

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Vec3f(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	ptRender->setColorMapRange(0, 5);

	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->statePointSet()->connect(ptRender->inPointSet());
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	colorMapper->outColor()->connect(ptRender->inColor());

	fluid->graphicsPipeline()->pushModule(calculateNorm);
	fluid->graphicsPipeline()->pushModule(colorMapper);
	fluid->graphicsPipeline()->pushModule(ptRender);

	//Create a container
	auto container = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
	container->loadCube(Vec3f(0.0f), Vec3f(1.0), 0.02, true);
	
	fluid->connect(container->importParticleSystems());

	return scn;
}

int main()
{
	GlfwApp window;
	window.setSceneGraph(createScene());
	window.createWindow(1280, 768);
	window.mainLoop();

	return 0;
}


