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

	

	BoxInfo box1, box2, box3, box4;
	RigidBodyInfo rigidBody;
	rigidBody.angularVelocity = Vec3f(60, 100, 60);
	Vec3f high = Vec3f(0, 1.0, 0);
	box1.center = Vec3f(0.0, 0.5, 0) + high;
	box1.halfLength = Vec3f(0.01, 0.2, 0.01);
	box1.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));

	box2.center = Vec3f(-0.21, 0.29, 0) + high;
	box2.halfLength = Vec3f(0.01, 0.2, 0.01);

	box3.center = Vec3f(0.21, 0.29, 0) + high;
	box3.halfLength = Vec3f(0.01, 0.2, 0.01);

	box4.center = Vec3f(0, 0.08, 0) + high;
	box4.halfLength = Vec3f(0.01, 0.2, 0.01);
	box4.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
	rigid->addBox(box1, rigidBody);
	rigid->addBox(box2, rigidBody);
	rigid->addBox(box3, rigidBody);
	rigid->addBox(box4, rigidBody);

	HingeJoint<Real> joint1(0, 1);
	joint1.setAnchorPoint(Vec3f(-0.21,0.5,0) + high, box1.center, box2.center, box1.rot, box2.rot);
	joint1.setAxis(Vec3f(0, 0, 1), box1.rot, box2.rot);
	joint1.setRange(-M_PI, M_PI);
	rigid->addHingeJoint(joint1);

	HingeJoint<Real> joint2(0, 2);
	joint2.setAnchorPoint(Vec3f(0.21, 0.5, 0) + high, box1.center, box3.center, box1.rot, box3.rot);
	joint2.setAxis(Vec3f(0, 0, 1), box1.rot, box3.rot);
	joint2.setRange(-M_PI, M_PI);
	rigid->addHingeJoint(joint2);

	HingeJoint<Real> joint3(1, 3);
	joint3.setAnchorPoint(Vec3f(-0.21, 0.08, 0) + high, box2.center, box4.center, box2.rot, box4.rot);
	joint3.setAxis(Vec3f(0, 0, 1), box2.rot, box4.rot);
	joint3.setRange(-M_PI, M_PI);
	rigid->addHingeJoint(joint3);

	HingeJoint<Real> joint4(2, 3);
	joint4.setAnchorPoint(Vec3f(0.21, 0.08, 0) + high, box3.center, box4.center, box3.rot, box4.rot);
	joint4.setAxis(Vec3f(0, 0, 1), box3.rot, box4.rot);
	joint4.setRange(-M_PI, M_PI);
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
	app.setSceneGraph(creatBricks());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


