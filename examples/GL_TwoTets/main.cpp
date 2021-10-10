#include <GlfwApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>

#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>

using namespace std;
using namespace dyno;

int main()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<RigidBodySystem<DataType3f>> rigid = scene.createNewScene<RigidBodySystem<DataType3f>>();

	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);

	TetInfo tet0;
	tet0.v[0] = Vec3f(0.5f, 0.1f, 0.5f);
	tet0.v[1] = Vec3f(0.5f, 0.2f, 0.5f);
	tet0.v[2] = Vec3f(0.6f, 0.1f, 0.5f);
	tet0.v[3] = Vec3f(0.5f, 0.1f, 0.6f);
	rigid->addTet(tet0, rigidBody);

	TetInfo tet1;
	tet1.v[0] = Vec3f(0.5f, 0.3f, 0.5f);
	tet1.v[1] = Vec3f(0.5f, 0.4f, 0.5f);
	tet1.v[2] = Vec3f(0.6f, 0.3f, 0.5f);
	tet1.v[3] = Vec3f(0.5f, 0.3f, 0.6f);
	rigid->addTet(tet1, rigidBody);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->currentTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(1, 1, 0));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	GLRenderEngine* engine = new GLRenderEngine;

	GlfwApp window;
	window.setRenderEngine(engine);
	window.createWindow(1280, 768);
	window.mainLoop();

	delete engine;

	return 0;
}


