#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Log.h>
#include <Peridynamics/ElasticBody.h>
#include <Peridynamics/Cloth.h>
#include <ParticleSystem/StaticBoundary.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto root = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
	root->loadCube(Vec3f(0), Vec3f(1), 0.005f, true);
	root->loadShpere(Vec3f(0.5, 0.7f, 0.5), 0.08f, 0.005f, false, true);

	auto cloth = scn->addNode(std::make_shared<Cloth<DataType3f>>());
	cloth->loadParticles(getAssetPath() + "cloth/cloth.obj");

	root->addParticleSystem(cloth);

	auto pointRenderer = std::make_shared<GLPointVisualModule>();
	pointRenderer->setColor(Vec3f(1, 0.2, 1));
	pointRenderer->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);
	cloth->statePointSet()->connect(pointRenderer->inPointSet());
	cloth->stateVelocity()->connect(pointRenderer->inColor());

	cloth->graphicsPipeline()->pushModule(pointRenderer);
	cloth->setVisible(true);

	auto surfaceRenderer = std::make_shared<GLSurfaceVisualModule>();
	cloth->statePointSet()->connect(surfaceRenderer->inTriangleSet());
	cloth->graphicsPipeline()->pushModule(surfaceRenderer);
	//cloth->getSurface()->graphicsPipeline()->pushPersistentModule(surfaceRenderer);

	return scn;
}

int main()
{
	GlfwApp window;
	window.setSceneGraph(createScene());
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}