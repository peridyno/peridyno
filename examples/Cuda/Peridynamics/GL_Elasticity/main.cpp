#include <UbiApp.h>

#include <SceneGraph.h>
#include <Peridynamics/ElasticBody.h>

#include <BasicShapes/CubeModel.h>

#include <Volume/BasicShapeToVolume.h>

#include <Multiphysics/VolumeBoundary.h>

// Internal OpenGL Renderer
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

// Modeling
#include <BasicShapes/CubeModel.h>

// ParticleSystem
#include <ParticleSystem/ParticleFluid.h>
#include "ParticleSystem/MakeParticleSystem.h"

#include <Samplers/CubeSampler.h>

using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	//Create a cube
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.6, 0.2, 0.5));
	cube->varLength()->setValue(Vec3f(0.1, 0.1, 0.1));
	cube->graphicsPipeline()->disable();

	//Create a sampler
	auto sampler = scn->addNode(std::make_shared<CubeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(0.005);
	sampler->graphicsPipeline()->disable();

	cube->outCube()->connect(sampler->inCube());

	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());

	sampler->statePointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto bunny = scn->addNode(std::make_shared<ElasticBody<DataType3f>>());

	initialParticles->connect(bunny->importSolidParticles());

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

	bunny->connect(container->importParticleSystems());

	auto pointRenderer = std::make_shared<GLPointVisualModule>();
	pointRenderer->varPointSize()->setValue(0.005);
	pointRenderer->setColor(Color(1, 0.2, 1));
	pointRenderer->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);
	bunny->statePointSet()->connect(pointRenderer->inPointSet());
	bunny->stateVelocity()->connect(pointRenderer->inColor());
	bunny->graphicsPipeline()->pushModule(pointRenderer);

	return scn;
}

int main()
{
	UbiApp app(GUIType::GUI_GLFW);
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}