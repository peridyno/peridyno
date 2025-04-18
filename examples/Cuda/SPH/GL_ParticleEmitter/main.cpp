#include <GlfwApp.h>

#include <SceneGraph.h>

#include <BasicShapes/CubeModel.h>

#include <Volume/BasicShapeToVolume.h>

#include <Multiphysics/VolumeBoundary.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/Emitters/SquareEmitter.h>

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
	emitter->connect(fluid->importParticleEmitters());

	//Create a container
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.5f));
	cubeBoundary->varLength()->setValue(Vec3f(1.0f));
	cubeBoundary->setVisible(false);

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.02f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());

	auto container = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	cube2vol->connect(container->importVolumes());
	
	fluid->connect(container->importParticleSystems());

	return scn;
}

int main()
{
	GlfwApp app;
	app.setSceneGraph(createScene());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


