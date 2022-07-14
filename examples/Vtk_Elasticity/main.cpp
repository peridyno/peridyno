#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Log.h>
#include <Peridynamics/ElasticBody.h>
#include <Peridynamics/Module/ElasticityModule.h>
#include <ParticleSystem/StaticBoundary.h>

// VTK Renderer
#include <VtkRenderEngine.h>
#include <VtkPointVisualModule.h>


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

	// point
	auto ptRender = std::make_shared<VtkPointVisualModule>();
	ptRender->setColor(1, 0, 0);
	bunny->statePointSet()->connect(ptRender->inPointSet());
	bunny->graphicsPipeline()->pushModule(ptRender);

	GlfwApp window;
	window.setRenderEngine(std::make_shared<VtkRenderEngine>());
	window.setSceneGraph(scn);

	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}


