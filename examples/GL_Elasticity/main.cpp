#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Peridynamics/ElasticBody.h>
#include <Peridynamics/ElasticityModule.h>
#include <ParticleSystem/StaticBoundary.h>

// Internal OpenGL Renderer
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>

using namespace dyno;

int main()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

 	auto root = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
 	root->loadCube(Vec3f(0), Vec3f(1), 0.005f, true);

	auto bunny = scn->addNode(std::make_shared<ElasticBody<DataType3f>>());
	root->addParticleSystem(bunny);

	bunny->loadParticles("../../data/bunny/bunny_points.obj");
	bunny->loadSurface("../../data/bunny/bunny_mesh.obj");
	bunny->scale(1.0f);
	bunny->translate(Vec3f(0.5f, 0.1f, 0.5f));
	bunny->setVisible(true);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(1, 1, 0));
	bunny->getSurfaceNode()->currentTopology()->connect(sRender->inTriangleSet());
	bunny->getSurfaceNode()->graphicsPipeline()->pushModule(sRender);

	GlfwApp window;
	window.setSceneGraph(scn);
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}


