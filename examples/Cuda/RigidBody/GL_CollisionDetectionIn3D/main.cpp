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
#include "Collision/CollistionDetectionBoundingBox.h"

using namespace std;
using namespace dyno;

void createTwoBoxes(std::shared_ptr<RigidBodySystem<DataType3f>> rigid)
{
	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	BoxInfo box;
	box.center = Vec3f(-0.3, 0.1, 0.5);
	box.halfLength = Vec3f(0.1, 0.1, 0.1);
	rigid->addBox(box, rigidBody);

	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	box.center = Vec3f(-0.3, 0.3, 0.59);
	box.halfLength = Vec3f(0.1, 0.1, 0.1);
	rigid->addBox(box, rigidBody);
}

void createTwoTets(std::shared_ptr<RigidBodySystem<DataType3f>> rigid)
{
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
}

void createTetBox(std::shared_ptr<RigidBodySystem<DataType3f>> rigid) {

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
}

void createTwoCapsules(std::shared_ptr<RigidBodySystem<DataType3f>> rigid) {
	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	CapsuleInfo capsule;
	capsule.center = Vec3f(-1.25, 0.1, -0.5f);
	capsule.rot = Quat1f(M_PI / 2, Vec3f(1, 0, 0));
	capsule.halfLength = 0.1f;
	capsule.radius = 0.1f;
	rigid->addCapsule(capsule, rigidBody);

	capsule.center = Vec3f(-1.3, 0.33, -0.5f);
	capsule.halfLength = 0.1f;
	capsule.rot = Quat1f(M_PI/2, Vec3f(1, 0, 0));
	capsule.radius = 0.1f;
	rigid->addCapsule(capsule, rigidBody);
}

void createCapsuleBox(std::shared_ptr<RigidBodySystem<DataType3f>> rigid) {
	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	BoxInfo box;
	box.center = Vec3f(-1.3, 0.1, 0.5);
	box.halfLength = Vec3f(0.1, 0.1, 0.1);
	rigid->addBox(box, rigidBody);

	CapsuleInfo capsule;
	capsule.center = Vec3f(-1.3, 0.3, 0.35);
	capsule.rot = Quat1f(M_PI / 2, Vec3f(1, 0, 0));
	capsule.halfLength = 0.1f;
	capsule.radius = 0.1f;
	rigid->addCapsule(capsule, rigidBody);
}

void createCapsuleTet(std::shared_ptr<RigidBodySystem<DataType3f>> rigid)
{
	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);

	TetInfo tet0;
	tet0.v[0] = Vec3f(0.45f, 0.0f, -0.45f);
	tet0.v[1] = Vec3f(0.45f, 0.25f, -0.45f);
	tet0.v[2] = Vec3f(0.7f, 0.0f, -0.45f);
	tet0.v[3] = Vec3f(0.45f, 0.0f, -0.2f);
	rigid->addTet(tet0, rigidBody);

	CapsuleInfo capsule;
	capsule.center = Vec3f(0.45, 0.4, -0.35);
	capsule.rot = Quat1f(M_PI / 2, Vec3f(1, 0, 0));
	capsule.halfLength = 0.1f;
	capsule.radius = 0.1f;
	rigid->addCapsule(capsule, rigidBody);
}

//create tests for sphere
void createTwoSpheres(std::shared_ptr<RigidBodySystem<DataType3f>> rigid)
{
	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	SphereInfo sphere;
	sphere.center = Vec3f(-1.3, 0.1, 1.5);
	sphere.radius = 0.1f;
	rigid->addSphere(sphere, rigidBody);

	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	sphere.center = Vec3f(-1.3, 0.3, 1.59);
	sphere.radius = 0.1f;
	rigid->addSphere(sphere, rigidBody);
}

void createSphereBox(std::shared_ptr<RigidBodySystem<DataType3f>> rigid)
{
	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	BoxInfo box;
	box.center = Vec3f(-0.3, 0.3, 1.5);
	box.halfLength = Vec3f(0.1f);
	rigid->addBox(box, rigidBody);

	SphereInfo sphere;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	sphere.center = Vec3f(-0.3, 0.1, 1.59);
	sphere.radius = 0.1f;
	rigid->addSphere(sphere, rigidBody);
}

void createSphereTet(std::shared_ptr<RigidBodySystem<DataType3f>> rigid)
{
	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	TetInfo tet;
	tet.v[0] = Vec3f(0.45f, 0.3f, 1.59);
	tet.v[1] = Vec3f(0.45f, 0.55f, 1.59);
	tet.v[2] = Vec3f(0.7f, 0.3f, 1.59);
	tet.v[3] = Vec3f(0.45f, 0.3f, 1.89);
	rigid->addTet(tet, rigidBody);

	SphereInfo sphere;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	sphere.center = Vec3f(0.7, 0.1, 1.59);
	sphere.radius = 0.1f;
	rigid->addSphere(sphere, rigidBody);
}

void createSphereCapsule(std::shared_ptr<RigidBodySystem<DataType3f>> rigid)
{
	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	CapsuleInfo cap;
	cap.center = Vec3f(1.3, 0.3, 1.6);
	cap.rot = Quat1f(M_PI / 2, Vec3f(1, 0, 0));
	cap.halfLength = 0.1f;
	cap.radius = 0.1f;
	rigid->addCapsule(cap, rigidBody);

	SphereInfo sphere;
	rigidBody.linearVelocity = Vec3f(0.0, 0, 0);
	sphere.center = Vec3f(1.3, 0.1, 1.59);
	sphere.radius = 0.1f;
	rigid->addSphere(sphere, rigidBody);
}

std::shared_ptr<SceneGraph> creatScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setAlpha(0.8f);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);


	//TODO: to enable using internal modules inside a node
	auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
	rigid->stateTopology()->connect(elementQuery->inDiscreteElements());
	rigid->stateCollisionMask()->connect(elementQuery->inCollisionMask());
	rigid->graphicsPipeline()->pushModule(elementQuery);

	//Visualize contact normals
	auto contactMapper = std::make_shared<ContactsToEdgeSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactMapper->inContacts());
	contactMapper->varScale()->setValue(0.02);
	rigid->graphicsPipeline()->pushModule(contactMapper);

	auto wireRender = std::make_shared<GLWireframeVisualModule>();
	wireRender->setColor(Color(0, 0, 1));
	contactMapper->outEdgeSet()->connect(wireRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(wireRender);

	//Visualize contact points
	auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(contactPointMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Color(1, 0, 0));
	pointRender->varPointSize()->setValue(0.01f);
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);

	//Visualize boundary contacts
		//Visualize contact points
	auto cdBV = std::make_shared<CollistionDetectionBoundingBox<DataType3f>>();
	rigid->stateTopology()->connect(cdBV->inDiscreteElements());
	//jeep->inTriangleSet()->connect(cdBV->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(cdBV);

	auto boundaryContactsMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	cdBV->outContacts()->connect(boundaryContactsMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(boundaryContactsMapper);

	auto boundaryContactsRender = std::make_shared<GLPointVisualModule>();
	boundaryContactsRender->setColor(Color(0, 1, 0));
	boundaryContactsRender->varPointSize()->setValue(0.01f);
	boundaryContactsMapper->outPointSet()->connect(boundaryContactsRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(boundaryContactsRender);

	createTwoBoxes(rigid);
	createTwoTets(rigid);
	createTetBox(rigid);
	createCapsuleBox(rigid);
	createTwoCapsules(rigid);
	createCapsuleTet(rigid);


	createTwoSpheres(rigid);
	createSphereBox(rigid);
	createSphereTet(rigid);
	createSphereCapsule(rigid);

	return scn;
}


int main()
{	
	GlfwApp app;
	app.setSceneGraph(creatScene());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


