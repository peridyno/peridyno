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

std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();


	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	BoxInfo box;
	box.center = Vec3f(0, 0.15, 0);
	box.halfLength = Vec3f(0.2, 0.05, 0.1);
	CapsuleInfo capsule1, capsule2, capsule3, capsule4;

	capsule1.center = Vec3f(-0.18, 0.03, 0.09);
	capsule1.halfLength = 0.01;
	capsule1.radius = 0.03;
	capsule2.center = Vec3f(-0.18, 0.03, -0.09);
	capsule2.halfLength = 0.01;
	capsule2.radius = 0.03;
	capsule3.center = Vec3f(0.18, 0.03, 0.09);
	capsule3.halfLength = 0.01;
	capsule3.radius = 0.03;
	capsule4.center = Vec3f(0.18, 0.03, -0.09);
	capsule4.halfLength = 0.01;
	capsule4.radius = 0.03;

	RigidBodyInfo rigidbody;

	rigid->addBox(box, rigidbody);
	rigidbody.angularVelocity = Vec3f(0, 0, 0);

	rigid->addCapsule(capsule1, rigidbody);
	rigid->addCapsule(capsule2, rigidbody);
	rigid->addCapsule(capsule3, rigidbody);
	rigid->addCapsule(capsule4, rigidbody);

	Real wheel_velocity = 40;

	HingeJoint<Real> joint1(1, 0);
	joint1.setAnchorPoint(capsule1.center, capsule1.center, box.center, capsule1.rot, box.rot);
	joint1.setMoter(-wheel_velocity);
	joint1.setAxis(Vec3f(0, 0, 1), capsule1.rot, box.rot);
	rigid->addHingeJoint(joint1);
	HingeJoint<Real> joint2(2, 0);
	joint2.setAnchorPoint(capsule2.center, capsule2.center, box.center, capsule2.rot, box.rot);
	joint2.setMoter(-wheel_velocity);
	joint2.setAxis(Vec3f(0, 0, 1), capsule2.rot, box.rot);
	rigid->addHingeJoint(joint2);
	HingeJoint<Real> joint3(3, 0);
	joint3.setAnchorPoint(capsule3.center, capsule3.center, box.center, capsule3.rot, box.rot);
	joint3.setMoter(-wheel_velocity);
	joint3.setAxis(Vec3f(0, 0, 1), capsule3.rot, box.rot);
	rigid->addHingeJoint(joint3);
	HingeJoint<Real> joint4(4, 0);
	joint4.setAnchorPoint(capsule4.center, capsule4.center, box.center, capsule4.rot, box.rot);
	joint4.setMoter(-wheel_velocity);
	joint4.setAxis(Vec3f(0, 0, 1), capsule4.rot, box.rot);
	rigid->addHingeJoint(joint4);
	






	

	



	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(0.3f, 0.5f, 0.9f));
	sRender->setAlpha(0.8f);
	sRender->setRoughness(0.7f);
	sRender->setMetallic(3.0f);
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
	contactMapper->varScale()->setValue(0.00002);
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
	pointRender->varPointSize()->setValue(0.00003f);
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);

	//Visualize Anchor point for joint
	auto anchorPointMapper = std::make_shared<AnchorPointToPointSet<DataType3f>>();
	rigid->stateCenter()->connect(anchorPointMapper->inCenter());
	rigid->stateRotationMatrix()->connect(anchorPointMapper->inRotationMatrix());
	rigid->stateBallAndSocketJoints()->connect(anchorPointMapper->inBallAndSocketJoints());
	rigid->stateSliderJoints()->connect(anchorPointMapper->inSliderJoints());
	//rigid->stateHingeJoints()->connect(anchorPointMapper->inHingeJoints());
	//rigid->stateFixedJoints()->connect(anchorPointMapper->inFixedJoints());
	rigid->graphicsPipeline()->pushModule(anchorPointMapper);

	auto pointRender2 = std::make_shared<GLPointVisualModule>();
	pointRender2->setColor(Color(1, 0, 0));
	pointRender2->varPointSize()->setValue(0.03f);
	anchorPointMapper->outPointSet()->connect(pointRender2->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender2);
	return scn;
}

int main()
{
	GlfwApp app;
	app.setSceneGraph(creatCar());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


