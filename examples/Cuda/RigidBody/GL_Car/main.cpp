#include <QtApp.h>

#include <SceneGraph.h>

#include <RigidBody/Vechicle.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>
#include <Mapping/AnchorPointToPointSet.h>

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionTriangleSet.h"
#include "Collision/CollistionDetectionBoundingBox.h"

#include <Module/GLPhotorealisticInstanceRender.h>

#include <PlaneModel.h>

#include "GltfLoader.h"


using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto jeep = scn->addNode(std::make_shared<Vechicle<DataType3f>>());

	auto prRender = std::make_shared<GLPhotorealisticInstanceRender>();
	jeep->inTextureMesh()->connect(prRender->inTextureMesh());
	jeep->stateInstanceTransform()->connect(prRender->inTransform());
	jeep->graphicsPipeline()->pushModule(prRender);

	BoxInfo box1, box2, box3, box4, box5, box6;
	box1.center = Vec3f(0, 1.171, -0.011);
	box1.halfLength = Vec3f(1.011, 0.5, 2.4);
	box2.center = Vec3f(0, 1.044, -2.254);
	box2.halfLength = Vec3f(0.250, 0.250, 0.250);

	box3.center = Vec3f(0.812, 1.171, 1.722);
	box3.halfLength = Vec3f(0.450, 0.2, 0.450);
	box4.center = Vec3f(-0.812, 1.171, 1.722);
	box4.halfLength = Vec3f(0.450, 0.2, 0.450);
	box5.center = Vec3f(-0.812, 1.171, -1.426);
	box5.halfLength = Vec3f(0.450, 0.2, 0.450);
	box6.center = Vec3f(0.812, 1.171, -1.426);
	box6.halfLength = Vec3f(0.450, 0.2, 0.450);
	CapsuleInfo capsule1, capsule2, capsule3, capsule4;

	capsule1.center = Vec3f(0.812, 0.450, 1.722);
	capsule1.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
	capsule1.halfLength = 0.1495;
	capsule1.radius = 0.450;
	capsule2.center = Vec3f(-0.812, 0.450, 1.722);
	capsule2.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
	capsule2.halfLength = 0.1495;
	capsule2.radius = 0.450;
	capsule3.center = Vec3f(-0.812, 0.450, -1.426);
	capsule3.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
	capsule3.halfLength = 0.1495;
	capsule3.radius = 0.450;
	capsule4.center = Vec3f(0.812, 0.450, -1.426);
	capsule4.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
	capsule4.halfLength = 0.1495;
	capsule4.radius = 0.450;

	RigidBodyInfo rigidbody;

	Vec3f offset = Vec3f(0.0f, -0.721f, 0.148f);
	rigidbody.offset = offset;
	jeep->addBox(box1, rigidbody, 100);

	rigidbody.offset = Vec3f(0.0f);

	jeep->addBox(box2, rigidbody, 100);
	jeep->addBox(box3, rigidbody, 100);
	jeep->addBox(box4, rigidbody, 100);
	jeep->addBox(box5, rigidbody, 100);
	jeep->addBox(box6, rigidbody, 100);

	Real wheel_velocity = 100;

	jeep->addCapsule(capsule1, rigidbody, 1);
	jeep->addCapsule(capsule2, rigidbody, 1);
	jeep->addCapsule(capsule3, rigidbody, 1);
	jeep->addCapsule(capsule4, rigidbody, 1);

	HingeJoint<Real> joint1(6, 2);
	joint1.setAnchorPoint(capsule1.center, capsule1.center, box3.center, capsule1.rot, box3.rot);
	joint1.setMoter(wheel_velocity);
	joint1.setAxis(Vec3f(1, 0, 0), capsule1.rot, box3.rot);
	jeep->addHingeJoint(joint1);
	HingeJoint<Real> joint2(7, 3);
	joint2.setAnchorPoint(capsule2.center, capsule2.center, box4.center, capsule2.rot, box4.rot);
	joint2.setMoter(wheel_velocity);
	joint2.setAxis(Vec3f(1, 0, 0), capsule2.rot, box4.rot);
	jeep->addHingeJoint(joint2);
	HingeJoint<Real> joint3(8, 4);
	joint3.setAnchorPoint(capsule3.center, capsule3.center, box5.center, capsule3.rot, box5.rot);
	joint3.setMoter(wheel_velocity);
	joint3.setAxis(Vec3f(1, 0, 0), capsule3.rot, box5.rot);
	jeep->addHingeJoint(joint3);
	HingeJoint<Real> joint4(9, 5);
	joint4.setAnchorPoint(capsule4.center, capsule4.center, box6.center, capsule4.rot, box6.rot);
	joint4.setMoter(wheel_velocity);
	joint4.setAxis(Vec3f(1, 0, 0), capsule4.rot, box6.rot);
	jeep->addHingeJoint(joint4);


	FixedJoint<Real> joint5(0, 1);
	joint5.setAnchorPoint((box1.center + offset + box2.center) / 2, box1.center + offset, box2.center, box1.rot, box2.rot);
	jeep->addFixedJoint(joint5);
	FixedJoint<Real> joint6(0, 2);
	joint6.setAnchorPoint((box1.center + offset + box3.center) / 2, box1.center + offset, box3.center, box1.rot, box3.rot);
	jeep->addFixedJoint(joint6);
	FixedJoint<Real> joint7(0, 3);
	joint7.setAnchorPoint((box1.center + offset + box4.center) / 2, box1.center + offset, box4.center, box1.rot, box4.rot);
	jeep->addFixedJoint(joint7);
	FixedJoint<Real> joint8(0, 4);
	joint8.setAnchorPoint((box1.center + offset + box5.center) / 2, box1.center + offset, box5.center, box1.rot, box5.rot);
	jeep->addFixedJoint(joint8);
	FixedJoint<Real> joint9(0, 5);
	joint9.setAnchorPoint((box1.center + offset + box6.center) / 2, box1.center + offset, box6.center, box1.rot, box6.rot);
	jeep->addFixedJoint(joint9);

	jeep->bind(0, Pair<uint, uint>(5, 0));
	jeep->bind(1, Pair<uint, uint>(4, 0));
	jeep->bind(6, Pair<uint, uint>(0, 0));
	jeep->bind(7, Pair<uint, uint>(1, 0));
	jeep->bind(8, Pair<uint, uint>(2, 0));
	jeep->bind(9, Pair<uint, uint>(3, 0));

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->setVisible(false);
	gltf->varFileName()->setValue(getAssetPath() + "Jeep/JeepGltf/jeep.gltf");

	gltf->stateTextureMesh()->connect(jeep->inTextureMesh());

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varScale()->setValue(Vec3f(10.0f));
	plane->stateTriangleSet()->connect(jeep->inTriangleSet());

// 	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
// 	jeep->stateTopology()->connect(mapper->inDiscreteElements());
// 	jeep->graphicsPipeline()->pushModule(mapper);
// 
// 	auto sRender = std::make_shared<GLSurfaceVisualModule>();
// 	sRender->setColor(Color(0.3f, 0.5f, 0.9f));
// 	sRender->setAlpha(0.8f);
// 	sRender->setRoughness(0.7f);
// 	sRender->setMetallic(3.0f);
// 	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
// 	jeep->graphicsPipeline()->pushModule(sRender);

	//TODO: to enable using internal modules inside a node
	//Visualize contact normals
	auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
	jeep->stateTopology()->connect(elementQuery->inDiscreteElements());
	jeep->stateCollisionMask()->connect(elementQuery->inCollisionMask());
	jeep->graphicsPipeline()->pushModule(elementQuery);

	auto contactMapper = std::make_shared<ContactsToEdgeSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactMapper->inContacts());
	contactMapper->varScale()->setValue(0.00002);
	jeep->graphicsPipeline()->pushModule(contactMapper);

	auto wireRender = std::make_shared<GLWireframeVisualModule>();
	wireRender->setColor(Color(0, 0, 1));
	contactMapper->outEdgeSet()->connect(wireRender->inEdgeSet());
	jeep->graphicsPipeline()->pushModule(wireRender);

// 	//Visualize contact points
// 	auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
// 	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
// 	jeep->graphicsPipeline()->pushModule(contactPointMapper);
// 
// 	auto pointRender = std::make_shared<GLPointVisualModule>();
// 	pointRender->setColor(Color(1, 0, 0));
// 	pointRender->varPointSize()->setValue(0.00003f);
// 	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
// 	jeep->graphicsPipeline()->pushModule(pointRender);

	//Visualize Anchor point for joint
	auto anchorPointMapper = std::make_shared<AnchorPointToPointSet<DataType3f>>();
	jeep->stateCenter()->connect(anchorPointMapper->inCenter());
	jeep->stateRotationMatrix()->connect(anchorPointMapper->inRotationMatrix());
	jeep->stateBallAndSocketJoints()->connect(anchorPointMapper->inBallAndSocketJoints());
	jeep->stateSliderJoints()->connect(anchorPointMapper->inSliderJoints());
	//rigid->stateHingeJoints()->connect(anchorPointMapper->inHingeJoints());
	//rigid->stateFixedJoints()->connect(anchorPointMapper->inFixedJoints());
	jeep->graphicsPipeline()->pushModule(anchorPointMapper);

	auto pointRender2 = std::make_shared<GLPointVisualModule>();
	pointRender2->setColor(Color(1, 0, 0));
	pointRender2->varPointSize()->setValue(0.03f);
	anchorPointMapper->outPointSet()->connect(pointRender2->inPointSet());
	jeep->graphicsPipeline()->pushModule(pointRender2);


// 	//Visualize contact points
// 	auto cdBV = std::make_shared<CollistionDetectionTriangleSet<DataType3f>>();
// 	jeep->stateTopology()->connect(cdBV->inDiscreteElements());
// 	jeep->inTriangleSet()->connect(cdBV->inTriangleSet());
// 	jeep->graphicsPipeline()->pushModule(cdBV);
// 
// 	auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
// 	cdBV->outContacts()->connect(contactPointMapper->inContacts());
// 	jeep->graphicsPipeline()->pushModule(contactPointMapper);
// 
// 	auto contactsRender = std::make_shared<GLPointVisualModule>();
// 	contactsRender->setColor(Color(1, 0, 0));
// 	contactsRender->varPointSize()->setValue(0.1f);
// 	contactPointMapper->outPointSet()->connect(contactsRender->inPointSet());
// 	jeep->graphicsPipeline()->pushModule(contactsRender);

	return scn;
}

int main()
{
	QtApp app;
	app.setSceneGraph(creatCar());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


