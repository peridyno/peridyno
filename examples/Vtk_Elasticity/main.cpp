#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Log.h>
#include <Peridynamics/ElasticBody.h>
#include <Peridynamics/ElasticityModule.h>
#include <ParticleSystem/StaticBoundary.h>

// VTK Renderer
#include <VtkRenderEngine.h>
#include <VtkSurfaceVisualModule.h>


using namespace dyno;

int main()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vec3f(0), Vec3f(1), 0.005f, true);

	std::shared_ptr<ElasticBody<DataType3f>> bunny = std::make_shared<ElasticBody<DataType3f>>();
	root->addParticleSystem(bunny);

	bunny->loadParticles("../../data/bunny/bunny_points.obj");
	bunny->loadSurface("../../data/bunny/bunny_mesh.obj");
	bunny->scale(1.0f);
	bunny->translate(Vec3f(0.5f, 0.1f, 0.5f));
	bunny->setVisible(true);

	bool useVTK = true;
	RenderEngine* engine;

	engine = new VtkRenderEngine;
	auto sRender = std::make_shared<VtkSurfaceVisualModule>();
	sRender->setColor(1, 1, 0);
	bunny->getSurfaceNode()->currentTopology()->connect(sRender->inTriangleSet());
	bunny->getSurfaceNode()->graphicsPipeline()->pushModule(sRender);

	GlfwApp window;
	window.setRenderEngine(engine);
	window.createWindow(1024, 768);
	window.mainLoop();

	delete engine;

	return 0;
}


