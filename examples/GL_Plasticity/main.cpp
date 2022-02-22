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
	auto boundary = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
	boundary->loadCube(Vec3f(0), Vec3f(1), 0.005, true);

	auto elastoplasticBody = scn->addNode(std::make_shared<ElastoplasticBody<DataType3f>>());
	elastoplasticBody->connect(boundary->importParticleSystems());

	elastoplasticBody->setVisible(false);
  	elastoplasticBody->loadParticles(Vec3f(-1.1), Vec3f(1.15), 0.1);
  	elastoplasticBody->loadSurface("../../data/standard/standard_cube20.obj");
	elastoplasticBody->scale(0.05);
	elastoplasticBody->translate(Vec3f(0.3, 0.2, 0.5));
	elastoplasticBody->getSurfaceNode()->setVisible(true);

	auto ptRender = std::make_shared<GLSurfaceVisualModule>();
	ptRender->setColor(Vec3f(0, 1, 1));
	elastoplasticBody->getSurfaceNode()->currentTopology()->connect(ptRender->inTriangleSet());
	elastoplasticBody->getSurfaceNode()->graphicsPipeline()->pushModule(ptRender);

	auto elasticBody = scn->addNode(std::make_shared<ElasticBody<DataType3f>>());
	boundary->addParticleSystem(elasticBody);

	elasticBody->setVisible(false);
	elasticBody->loadParticles(Vec3f(-1.1), Vec3f(1.15), 0.1);
	elasticBody->loadSurface("../../data/standard/standard_cube20.obj");
	elasticBody->scale(0.05);
	elasticBody->translate(Vec3f(0.5, 0.2, 0.5));

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(1, 1, 1));
	elasticBody->getSurfaceNode()->currentTopology()->connect(sRender->inTriangleSet());
	elasticBody->getSurfaceNode()->graphicsPipeline()->pushModule(sRender);

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


