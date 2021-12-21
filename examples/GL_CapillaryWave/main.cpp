#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Log.h>
#include <Peridynamics/ElasticBody.h>
#include <HeightField/CapillaryWave.h>
#include <ParticleSystem/StaticBoundary.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	//root->loadCube(Vec3f(0), Vec3f(10), 0.005f, true);
	root->loadShpere(Vec3f(0.5, 0.7f, 0.5), 0.08f, 0.005f, false, true);

	std::shared_ptr<CapillaryWave<DataType3f>> cloth = std::make_shared<CapillaryWave<DataType3f>>();
	cloth->loadParticles("../../data/cloth/cloth1.obj");
	//cloth->loadSurface("../../data/cloth/cloth1.obj");

	//root->addParticleSystem(cloth);
	/*
	auto pointRenderer = std::make_shared<GLPointVisualModule>();
	pointRenderer->setColor(Vec3f(1, 0.2, 1));
	pointRenderer->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);
	cloth->currentTopology()->connect(pointRenderer->inPointSet());
	cloth->currentVelocity()->connect(pointRenderer->inColor());

	cloth->graphicsPipeline()->pushModule(pointRenderer);
	cloth->setVisible(true);

	auto surfaceRenderer = std::make_shared<GLSurfaceVisualModule>();
	cloth->currentTopology()->connect(surfaceRenderer->inTriangleSet());
	cloth->graphicsPipeline()->pushModule(surfaceRenderer);
	//cloth->getSurface()->graphicsPipeline()->pushPersistentModule(surfaceRenderer);
	*/
}

int main()
{
	CreateScene();

	RenderEngine* engine = new GLRenderEngine;
	
	GlfwApp window;
	window.setRenderEngine(engine);
	window.createWindow(1024, 768);
	window.mainLoop();

	delete engine;

	return 0;
}