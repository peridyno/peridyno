#include <GlfwApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>
#include <Mapping/AnchorPointToPointSet.h>

#include "Collision/NeighborElementQuery.h"


using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatBricks()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();


	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	

	RigidBodyInfo rigidbody;
	BoxInfo box1, box2, box3, box4;

	box1.center = Vec3f(0, 0.8, 0);
	box1.halfLength = Vec3f(0.03, 0.2, 0.03);
	box1.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));

	box2.center = Vec3f(0, 0.3, 0);
	box2.halfLength = Vec3f(0.03, 0.2, 0.03);
	box2.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));

	box3.center = Vec3f(-0.26, 0.55, 0);
	box3.halfLength = Vec3f(0.03, 0.2, 0.03);

	box4.center = Vec3f(0.26, 0.55, 0);
	box4.halfLength = Vec3f(0.03, 0.2, 0.03);
	rigidbody.angularVelocity = Vec3f(60, 200, 60);
	auto boxActor1 = rigid->addBox(box1, rigidbody);
	auto boxActor2 = rigid->addBox(box2, rigidbody);
	auto boxActor3 = rigid->addBox(box3, rigidbody);
	auto boxActor4 = rigid->addBox(box4, rigidbody);


	auto& hingeJoint1 = rigid->createHingeJoint(boxActor1, boxActor3);
	hingeJoint1.setAnchorPoint(Vec3f(-0.26, 0.8, 0));
	hingeJoint1.setAxis(Vec3f(0, 0, 1));
	hingeJoint1.setRange(-M_PI * 2 / 3, M_PI * 2 / 3);
	auto& hingeJoint2 = rigid->createHingeJoint(boxActor1, boxActor4);
	hingeJoint2.setAnchorPoint(Vec3f(0.26, 0.8, 0));
	hingeJoint2.setAxis(Vec3f(0, 0, 1));
	hingeJoint2.setRange(-M_PI * 2 / 3, M_PI * 2 / 3);
	auto& hingeJoint3 = rigid->createHingeJoint(boxActor2, boxActor3);
	hingeJoint3.setAnchorPoint(Vec3f(-0.26, 0.3, 0));
	hingeJoint3.setAxis(Vec3f(0, 0, 1));
	hingeJoint3.setRange(-M_PI * 2 / 3, M_PI * 2 / 3);
	auto& hingeJoint4 = rigid->createHingeJoint(boxActor2, boxActor4);
	hingeJoint4.setAnchorPoint(Vec3f(0.26, 0.3, 0));
	hingeJoint4.setAxis(Vec3f(0, 0, 1));
	hingeJoint4.setRange(-M_PI * 2 / 3, M_PI * 2 / 3);
	
	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(1, 1, 0));
	sRender->setAlpha(1.0f);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	//TODO: to enable using internal modules inside a node
	//Visualize contact normals
	auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
	rigid->stateTopology()->connect(elementQuery->inDiscreteElements());
	rigid->stateCollisionMask()->connect(elementQuery->inCollisionMask());
	rigid->graphicsPipeline()->pushModule(elementQuery);

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
	pointRender->varPointSize()->setValue(0.003f);
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);

	return scn;
}

int main()
{
	GlfwApp app;
	app.setSceneGraph(creatBricks());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


