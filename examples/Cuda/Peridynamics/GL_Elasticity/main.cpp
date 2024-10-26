#include <UbiApp.h>

#include <SceneGraph.h>
#include <Peridynamics/ElasticBody.h>

#include <ParticleSystem/StaticBoundary.h>

// Internal OpenGL Renderer
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

// Modeling
#include <BasicShapes/CubeModel.h>

// ParticleSystem
#include <ParticleSystem/ParticleFluid.h>
#include "ParticleSystem/MakeParticleSystem.h"
#include <ParticleSystem/CubeSampler.h>

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

	auto boundary = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
	boundary->loadCube(Vec3f(0), Vec3f(1), 0.005f, true);
	bunny->connect(boundary->importParticleSystems());

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