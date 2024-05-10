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

	RigidBodyInfo rigidBody;
	BoxInfo box1, box2, box3, box4, box5;
	rigidBody.linearVelocity = Vec3f(0, 0, 0);
	
	box1.center = Vec3f(0, 0.3, 0);
	box1.halfLength = Vec3f(0.1, 0.2, 0.1);
	rigid->addBox(box1, rigidBody);

	rigidBody.linearVelocity = Vec3f(0, 0, 0);


	box2.center = Vec3f(0, 0.05, 0);
	box2.halfLength = Vec3f(1, 0.05, 0.3);
	rigid->addBox(box2, rigidBody);

	rigidBody.linearVelocity = Vec3f(-10, 0, 0);

	box3.center = Vec3f(0.5, 0.2, 0);
	box3.halfLength = Vec3f(0.1, 0.1, 0.1);

	rigid->addBox(box3, rigidBody);


	SliderJoint<Real> joint1(0, 1);
	Vec3f point = Vec3f(0, 0.1, 0);
	joint1.setAnchorPoint(point, box1.center, box2.center, box1.rot, box2.rot);
	joint1.setAxis(Vec3f(1, 0, 0));
	joint1.setRange(-0.9, 0.9);
	rigid->addSliderJoint(joint1);

	rigidBody.linearVelocity = Vec3f(0, 0, 0);

	box4.center = Vec3f(0, 0.15, -0.2);
	box4.halfLength = Vec3f(1, 0.05, 0.1);
	rigid->addBox(box4, rigidBody);
	
	box5.center = Vec3f(0, 0.15, 0.2);
	box5.halfLength = Vec3f(1, 0.05, 0.1);
	rigid->addBox(box5, rigidBody);

	FixedJoint<Real> joint2(1, 3);
	point = Vec3f(0, 0.1, -0.2);
	joint2.setAnchorPoint(point, box2.center, box4.center, box2.rot, box5.rot);
	rigid->addFixedJoint(joint2);
	FixedJoint<Real> joint3(1, 4);
	point = Vec3f(0, 0.1, 0.2);
	joint3.setAnchorPoint(point, box2.center, box5.center, box2.rot, box5.rot);
    rigid->addFixedJoint(joint3);



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

	//Visualize Anchor point for joint
// 	auto anchorPointMapper = std::make_shared<AnchorPointToPointSet<DataType3f>>();
// 	rigid->stateCenter()->connect(anchorPointMapper->inCenter());
// 	rigid->stateRotationMatrix()->connect(anchorPointMapper->inRotationMatrix());
// 	rigid->stateBallAndSocketJoints()->connect(anchorPointMapper->inBallAndSocketJoints());
// 	rigid->stateSliderJoints()->connect(anchorPointMapper->inSliderJoints());
// 	//rigid->stateHingeJoints()->connect(anchorPointMapper->inHingeJoints());
// 	//rigid->stateFixedJoints()->connect(anchorPointMapper->inFixedJoints());
// 	rigid->graphicsPipeline()->pushModule(anchorPointMapper);
// 
// 	auto pointRender2 = std::make_shared<GLPointVisualModule>();
// 	pointRender2->setColor(Color(1, 0, 0));
// 	pointRender2->varPointSize()->setValue(0.01f);
// 	anchorPointMapper->outPointSet()->connect(pointRender2->inPointSet());
// 	rigid->graphicsPipeline()->pushModule(pointRender2);

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


