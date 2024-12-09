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

#include "Collision/NeighborElementQuery.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatBricks()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());
	RigidBodyInfo rA, rB, rC, rD;
	BoxInfo box1, box2, box3;

	rA.position = Vec3f(0, 0.5, 0);
	box1.halfLength = Vec3f(1.0, 0.05, 1.0);

	rB.position = Vec3f(0, 0.2, 0.8);
	box2.halfLength = Vec3f(0.7, 0.1, 0.1);

	rC.position = Vec3f(0, 0.2, -0.8);
	box3.halfLength = Vec3f(0.7, 0.1, 0.1);

	auto bodyActor = rigid->addBox(box1, rA);
	auto frontActor = rigid->addBox(box2, rB);
	auto rearActor = rigid->addBox(box3, rC);

	SphereInfo sphere;
	sphere.radius = 0.1;

	rA.position = Vec3f(0.9, 0.1, 0.8);
	
	rB.position = Vec3f(-0.9, 0.1, 0.8);

	rC.position = Vec3f(0.9, 0.1, -0.8);

	rD.position = Vec3f(-0.9, 0.1, -0.8);

	auto frontLeftTire = rigid->addSphere(sphere, rA);
	auto frontRightTire = rigid->addSphere(sphere, rB);
	auto rearLeftTire = rigid->addSphere(sphere, rC);
	auto rearRightTire = rigid->addSphere(sphere, rD);

	auto& joint1 = rigid->createHingeJoint(frontLeftTire, frontActor);
	joint1.setAnchorPoint(frontLeftTire->center);
	joint1.setAxis(Vec3f(1, 0, 0));
	joint1.setMoter(30);

	auto& joint2 = rigid->createHingeJoint(frontRightTire, frontActor);
	joint2.setAnchorPoint(frontRightTire->center);
	joint2.setAxis(Vec3f(1, 0, 0));
	joint2.setMoter(30);

	auto& joint3 = rigid->createHingeJoint(rearLeftTire, rearActor);
	joint3.setAnchorPoint(rearLeftTire->center);
	joint3.setAxis(Vec3f(1, 0, 0));
	joint3.setMoter(30);

	auto& joint4 = rigid->createHingeJoint(rearRightTire, rearActor);
	joint4.setAnchorPoint(rearRightTire->center);
	joint4.setAxis(Vec3f(1, 0, 0));
	joint4.setMoter(30);

	auto& joint5 = rigid->createFixedJoint(rearActor, bodyActor);
	joint5.setAnchorPoint(rearActor->center);

	auto& joint6 = rigid->createHingeJoint(frontActor, bodyActor);
	joint6.setAnchorPoint(frontActor->center);
	joint6.setAxis(Vec3f(0, 1, 0));
	joint6.setRange(M_PI / 12, M_PI / 12);


	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(0.204, 0.424, 0.612));
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
	wireRender->setColor(Color(0, 0, 0));
	mapper->outTriangleSet()->connect(wireRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(wireRender);

	//Visualize contact points
	/*auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(contactPointMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Color(1, 0, 0));
	pointRender->varPointSize()->setValue(0.003f);
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);*/

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


