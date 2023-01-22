#include <GlfwApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>

#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>
#include <GLPointVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include "Collision/NeighborElementQuery.h"

using namespace std;
using namespace dyno;

void createTwoBoxes(std::shared_ptr<SceneGraph>& scn) {
	{
		auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

		RigidBodyInfo rigidBody;
		rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
		BoxInfo box;
		box.center = Vec3f(-0.3, 0.1, 0.5);
		box.halfLength = Vec3f(0.1, 0.1, 0.1);
		rigid->addBox(box, rigidBody);

		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		rigid->stateTopology()->connect(mapper->inDiscreteElements());
		rigid->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->setColor(Vec3f(1, 1, 0));
		sRender->setAlpha(0.8f);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		rigid->graphicsPipeline()->pushModule(sRender);

	}

	{
		auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

		RigidBodyInfo rigidBody;
		rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
		BoxInfo box;
		box.center = Vec3f(-0.3, 0.3, 0.59);
		box.halfLength = Vec3f(0.1, 0.1, 0.1);
		rigid->addBox(box, rigidBody);

		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		rigid->stateTopology()->connect(mapper->inDiscreteElements());
		rigid->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->setColor(Vec3f(0, 1, 1));
		sRender->setAlpha(0.8f);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		rigid->graphicsPipeline()->pushModule(sRender);
	}
}

void createTwoTets(std::shared_ptr<SceneGraph>& scn) {
	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	
	TetInfo tet0;
	tet0.v[0] = Vec3f(0.45f, 0.3f, 0.45f);
	tet0.v[1] = Vec3f(0.45f, 0.55f, 0.45f);
	tet0.v[2] = Vec3f(0.7f, 0.3f, 0.45f);
	tet0.v[3] = Vec3f(0.45f, 0.3f, 0.7f);
	rigid->addTet(tet0, rigidBody);

	TetInfo tet1;
	tet1.v[0] = Vec3f(0.45f, 0.0f, 0.45f);
	tet1.v[1] = Vec3f(0.45f, 0.25f, 0.45f);
	tet1.v[2] = Vec3f(0.7f, 0.0f, 0.45f);
	tet1.v[3] = Vec3f(0.45f, 0.0f, 0.7f);
	rigid->addTet(tet1, rigidBody);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(1, 1, 0));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	//TODO: to enable using internal modules inside a node
	//Visualize contact normals
	auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
	rigid->stateTopology()->connect(elementQuery->inDiscreteElements());
	rigid->stateCollisionMask()->connect(elementQuery->inCollisionMask());
	rigid->graphicsPipeline()->pushModule(elementQuery);

	//Visualize contact points
	auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(contactPointMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Vec3f(1, 0, 0));
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);
}

void createTetBox(std::shared_ptr<SceneGraph>& scn) {
	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	BoxInfo box;
	box.center = Vec3f(1.3, 0.1, 0.5);
	box.halfLength = Vec3f(0.1, 0.1, 0.1);
	rigid->addBox(box, rigidBody);

	TetInfo tet0;
	tet0.v[0] = Vec3f(1.25f, 0.25f, 0.45f);
	tet0.v[1] = Vec3f(1.25f, 0.5f, 0.45f);
	tet0.v[2] = Vec3f(1.5f, 0.25f, 0.45f);
	tet0.v[3] = Vec3f(1.25f, 0.25f, 0.7f);
	rigid->addTet(tet0, rigidBody);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(1, 1, 0));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);


	//TODO: to enable using internal modules inside a node
	//Visualize contact normals
	auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
	rigid->stateTopology()->connect(elementQuery->inDiscreteElements());
	rigid->stateCollisionMask()->connect(elementQuery->inCollisionMask());
	rigid->graphicsPipeline()->pushModule(elementQuery);

	//Visualize contact points
	auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(contactPointMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Vec3f(1, 0, 0));
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);

}

std::shared_ptr<SceneGraph> creatScene(){
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	createTwoBoxes(scn);
	createTwoTets(scn);
	createTetBox(scn);
	return scn;
}


int main()
{	
	GlfwApp app;
	app.setSceneGraph(creatScene());
	app.createWindow(1280, 768);
	app.mainLoop();

	return 0;
}


