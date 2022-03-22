#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Log.h>
#include <Peridynamics/ElasticBody.h>
#include <Peridynamics/Module/ElasticityModule.h>
#include <ParticleSystem/StaticBoundary.h>

// VTK Renderer
#include <VtkRenderEngine.h>
#include <VtkSurfaceVisualModule.h>


using namespace dyno;

int main()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto root = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
	root->loadCube(Vec3f(0), Vec3f(1), 0.005f, true);

	auto bunny = scn->addNode(std::make_shared<ElasticBody<DataType3f>>());
	root->addParticleSystem(bunny);

	//TODO: fix the compilation errors
// 	bunny->loadParticles("../../data/bunny/bunny_points.obj");
// 	bunny->loadSurface("../../data/bunny/bunny_mesh.obj");
// 	bunny->scale(1.0f);
// 	bunny->translate(Vec3f(0.5f, 0.1f, 0.5f));
// 	bunny->setVisible(true);
// 
// 	bool useVTK = true;
// 	auto sRender = std::make_shared<VtkSurfaceVisualModule>();
// 	sRender->setColor(1, 1, 0);
// 	bunny->getSurfaceNode()->stateTopology()->connect(sRender->inTriangleSet());
// 	bunny->getSurfaceNode()->graphicsPipeline()->pushModule(sRender);

	GlfwApp window;
	window.setRenderEngine(std::make_shared<VtkRenderEngine>());
	window.setSceneGraph(scn);

	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}


