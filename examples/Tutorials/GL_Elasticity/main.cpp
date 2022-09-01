#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Peridynamics/ElasticBody.h>

#include <ParticleSystem/StaticBoundary.h>

// Internal OpenGL Renderer
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

using namespace dyno;

int main()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

 	auto root = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
 	root->loadCube(Vec3f(0), Vec3f(1), 0.005f, true);

	auto bunny = scn->addNode(std::make_shared<ElasticBody<DataType3f>>());
	bunny->connect(root->importParticleSystems());

	bunny->loadParticles(getAssetPath() + "bunny/bunny_points.obj");
	bunny->scale(1.0f);
	bunny->translate(Vec3f(0.5f, 0.1f, 0.5f));
	bunny->setVisible(true);

	auto pointRenderer = std::make_shared<GLPointVisualModule>();
	pointRenderer->setColor(Vec3f(1, 0.2, 1));
	pointRenderer->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);
	bunny->statePointSet()->connect(pointRenderer->inPointSet());
	bunny->stateVelocity()->connect(pointRenderer->inColor());
	bunny->graphicsPipeline()->pushModule(pointRenderer);

	GlfwApp window;
	window.setSceneGraph(scn);
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}


