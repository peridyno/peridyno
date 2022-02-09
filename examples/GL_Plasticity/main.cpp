#include <GlfwApp.h>

#include <SceneGraph.h>
#include <Log.h>
#include <Topology/PointSet.h>

#include <ParticleSystem/StaticBoundary.h>

#include <Peridynamics/ElastoplasticBody.h>
#include <Peridynamics/ElasticBody.h>
#include <Peridynamics/ElasticityModule.h>

#include <RigidBody/RigidBody.h>

#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	auto root = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
	root->loadCube(Vec3f(0), Vec3f(1), 0.005, true);

	auto child3 = scn->addNode(std::make_shared<ElastoplasticBody<DataType3f>>());
	root->addParticleSystem(child3);

	child3->setVisible(false);
  	child3->loadParticles(Vec3f(-1.1), Vec3f(1.15), 0.1);
  	child3->loadSurface("../../data/standard/standard_cube20.obj");
	child3->scale(0.05);
	child3->translate(Vec3f(0.3, 0.2, 0.5));
	child3->getSurfaceNode()->setVisible(true);

	auto ptRender = std::make_shared<GLSurfaceVisualModule>();
	ptRender->setColor(Vec3f(0, 1, 1));
	child3->getSurfaceNode()->currentTopology()->connect(ptRender->inTriangleSet());
	child3->getSurfaceNode()->graphicsPipeline()->pushModule(ptRender);

	auto child2 = scn->addNode(std::make_shared<ElasticBody<DataType3f>>());
	root->addParticleSystem(child2);

	child2->setVisible(false);
	child2->loadParticles(Vec3f(-1.1), Vec3f(1.15), 0.1);
	child2->loadSurface("../../data/standard/standard_cube20.obj");
	child2->scale(0.05);
	child2->translate(Vec3f(0.5, 0.2, 0.5));

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(1, 1, 1));
	child2->getSurfaceNode()->currentTopology()->connect(sRender->inTriangleSet());
	child2->getSurfaceNode()->graphicsPipeline()->pushModule(sRender);

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


