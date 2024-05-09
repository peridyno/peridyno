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

	uint N = 2;

	for (uint i = 0; i < N; i++)
	{
		Vec3f tr = i * Vec3f(3.0f, 0.0f, 0.0f);

		BoxInfo box1, box2, box3, box4;
		//car body
		box1.center = Vec3f(0, 1.171, -0.011) + tr;
		box1.halfLength = Vec3f(1.011, 0.793, 2.4);


		box2.center = Vec3f(0, 1.044, -2.254) + tr;
		box2.halfLength = Vec3f(0.447, 0.447, 0.15);

		box3.center = Vec3f(0.812, 0.450, 1.722) + tr;
		box3.halfLength = Vec3f(0.2f);
		box4.center = Vec3f(-0.812, 0.450, 1.722) + tr;
		box4.halfLength = Vec3f(0.2f);
		CapsuleInfo capsule1, capsule2, capsule3, capsule4;

		capsule1.center = Vec3f(0.812, 0.450, 1.722) + tr;
		capsule1.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
		capsule1.halfLength = 0.1495;
		capsule1.radius = 0.450;
		capsule2.center = Vec3f(-0.812, 0.450, 1.722) + tr;
		capsule2.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
		capsule2.halfLength = 0.1495;
		capsule2.radius = 0.450;
		capsule3.center = Vec3f(-0.812, 0.450, -1.426) + tr;
		capsule3.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
		capsule3.halfLength = 0.1495;
		capsule3.radius = 0.450;
		capsule4.center = Vec3f(0.812, 0.450, -1.426) + tr;
		capsule4.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
		capsule4.halfLength = 0.1495;
		capsule4.radius = 0.450;

		RigidBodyInfo rigidbody;

		rigidbody.bodyId = i;

		Vec3f offset = Vec3f(0.0f, -0.721f, 0.148f);
		rigidbody.offset = offset;
		auto bodyActor = jeep->addBox(box1, rigidbody, 10);

		rigidbody.offset = Vec3f(0.0f);

		auto spareTireActor = jeep->addBox(box2, rigidbody, 100);
		auto frontLeftSteerActor = jeep->addBox(box3, rigidbody, 1000);
		auto frontRightSteerActor = jeep->addBox(box4, rigidbody, 1000);

		Real wheel_velocity = 10;

		auto frontLeftTireActor = jeep->addCapsule(capsule1, rigidbody, 100);
		auto frontRightTireActor = jeep->addCapsule(capsule2, rigidbody, 100);
		auto rearLeftTireActor = jeep->addCapsule(capsule3, rigidbody, 100);
		auto rearRightTireActor = jeep->addCapsule(capsule4, rigidbody, 100);

		//front rear
		auto& joint1 = jeep->createHingeJoint(frontLeftTireActor, frontLeftSteerActor);
		joint1.setAnchorPoint(frontLeftTireActor->center, frontLeftTireActor->center, frontLeftSteerActor->center, frontLeftTireActor->rot, frontLeftSteerActor->rot);
		joint1.setMoter(wheel_velocity);
		joint1.setAxis(Vec3f(1, 0, 0), frontLeftTireActor->rot, frontLeftSteerActor->rot);

		auto& joint2 = jeep->createHingeJoint(frontRightTireActor, frontRightSteerActor);
		joint2.setAnchorPoint(frontRightTireActor->center, frontRightTireActor->center, frontRightSteerActor->center, frontRightTireActor->rot, frontRightSteerActor->rot);
		joint2.setMoter(wheel_velocity);
		joint2.setAxis(Vec3f(1, 0, 0), frontRightTireActor->rot, frontRightSteerActor->rot);

		//back rear
		auto& joint3 = jeep->createHingeJoint(rearLeftTireActor, bodyActor);
		joint3.setAnchorPoint(rearLeftTireActor->center, rearLeftTireActor->center, bodyActor->center, rearLeftTireActor->rot, bodyActor->rot);
		joint3.setMoter(wheel_velocity);
		joint3.setAxis(Vec3f(1, 0, 0), rearLeftTireActor->rot, bodyActor->rot);

		auto& joint4 = jeep->createHingeJoint(rearRightTireActor, bodyActor);
		joint4.setAnchorPoint(rearRightTireActor->center, rearRightTireActor->center, bodyActor->center, rearRightTireActor->rot, bodyActor->rot);
		joint4.setMoter(wheel_velocity);
		joint4.setAxis(Vec3f(1, 0, 0), rearRightTireActor->rot, bodyActor->rot);


		//FixedJoint<Real> joint5(0, 1);
		auto& joint5 = jeep->createFixedJoint(bodyActor, spareTireActor);
		joint5.setAnchorPoint((bodyActor->center + spareTireActor->center) / 2, bodyActor->center, spareTireActor->center, bodyActor->rot, spareTireActor->rot);
		auto& joint6 = jeep->createFixedJoint(bodyActor, frontLeftSteerActor);
		joint6.setAnchorPoint((bodyActor->center + frontLeftSteerActor->center) / 2, bodyActor->center, frontLeftSteerActor->center, bodyActor->rot, frontLeftSteerActor->rot);
		auto& joint7 = jeep->createFixedJoint(bodyActor, frontRightSteerActor);
		joint7.setAnchorPoint((bodyActor->center + frontRightSteerActor->center) / 2, bodyActor->center, frontRightSteerActor->center, bodyActor->rot, frontRightSteerActor->rot);

		jeep->bind(bodyActor, Pair<uint, uint>(5, i));
		jeep->bind(spareTireActor, Pair<uint, uint>(4, i));
		jeep->bind(frontLeftTireActor, Pair<uint, uint>(0, i));
		jeep->bind(frontRightTireActor, Pair<uint, uint>(1, i));
		jeep->bind(rearLeftTireActor, Pair<uint, uint>(2, i));
		jeep->bind(rearRightTireActor, Pair<uint, uint>(3, i));
	}

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->setVisible(false);
	gltf->varFileName()->setValue(getAssetPath() + "Jeep/JeepGltf/jeep.gltf");

	gltf->stateTextureMesh()->connect(jeep->inTextureMesh());

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varScale()->setValue(Vec3f(100.0f));
	plane->stateTriangleSet()->connect(jeep->inTriangleSet());

	//Visualize rigid bodies
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

	//Visualize Anchor point for joint
	auto anchorPointMapper = std::make_shared<AnchorPointToPointSet<DataType3f>>();
	jeep->stateCenter()->connect(anchorPointMapper->inCenter());
	jeep->stateRotationMatrix()->connect(anchorPointMapper->inRotationMatrix());
	jeep->stateBallAndSocketJoints()->connect(anchorPointMapper->inBallAndSocketJoints());
	jeep->stateSliderJoints()->connect(anchorPointMapper->inSliderJoints());
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


