#include <GlfwApp.h>

#include <SceneGraph.h>

#include "H_RigidBodySystem.h"

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include "Collision/NeighborElementQuery.h"

using namespace std;
using namespace dyno;

void creat_scene_fluid()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<RigidBodySystem<DataType3f>> rigid = scene.createNewScene<RigidBodySystem<DataType3f>>();

	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0, 0, 0);
	BoxInfo box;

	box.center = Vec3f(1.0f, 1.0f, 0.5f);
	box.halfLength = 0.5f * Vec3f(0.75f, 0.1f, 0.1f);
	box.rot = Quat<Real>(M_PI/4.0f,Vec3f(0.0f,0.0f,1.0f));
	rigid->addBox(box, rigidBody);

	box.center = Vec3f(1.0f, 1.6f, 0.5f);
	box.rot = Quat<Real>(-M_PI / 4.0f, Vec3f(0.0f, 0.0f, 1.0f));
	rigid->addBox(box, rigidBody);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->currentTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(1, 1, 0));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	//TODO: to enable using internal modules inside a node
	//Visualize contact normals
	auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
	rigid->currentTopology()->connect(elementQuery->inDiscreteElements());
	rigid->stateCollisionMask()->connect(elementQuery->inCollisionMask());
	rigid->graphicsPipeline()->pushModule(elementQuery);

	auto contactMapper = std::make_shared<ContactsToEdgeSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactMapper->inContacts());
	contactMapper->varScale()->setValue(0.02);
	rigid->graphicsPipeline()->pushModule(contactMapper);

	auto wireRender = std::make_shared<GLWireframeVisualModule>();
	wireRender->setColor(Vec3f(0, 1, 0));
	contactMapper->outEdgeSet()->connect(wireRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(wireRender);

	//Visualize contact points
	auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(contactPointMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Vec3f(1, 0, 0));
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);

	GLRenderEngine* engine = new GLRenderEngine;

	GlfwApp window;
	window.setRenderEngine(engine);
	window.createWindow(1280, 768);
	window.mainLoop();

	delete engine;
}

int main()
{
	creat_scene_fluid();
	//test_render();

	return 0;
}


