#include <QtApp.h>

#include <SceneGraph.h>

#include <RigidBody/ArticulatedBody.h>
#include <RigidBody/MultibodySystem.h>

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

#include <BasicShapes/PlaneModel.h>

#include "GltfLoader.h"


using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto jeep = scn->addNode(std::make_shared<ArticulatedBody<DataType3f>>());
	jeep->varFilePath()->setValue(getAssetPath() + "Jeep/JeepGltf/jeep.gltf");

	uint N = 1;

	for (uint i = 0; i < N; i++)
	{
		Vec3f tr = i * Vec3f(3.0f, 0.0f, 0.0f);

		BoxInfo box1, box2, box3, box4, box5, box6;
		//car body
		//box1.center = Vec3f(0, 1.171, -0.011) + tr;
		box1.halfLength = Vec3f(1.011, 0.793, 2.4);

		//box2.center = Vec3f(0, 1.044, -2.254) + tr;
		box2.halfLength = Vec3f(0.447, 0.447, 0.15);

		box3.center = Vec3f(0.812, 0.450, 1.722) + tr;
		box3.halfLength = Vec3f(0.2f);
		box4.center = Vec3f(-0.812, 0.450, 1.722) + tr;
		box4.halfLength = Vec3f(0.2f);

		SphereInfo sphere1, sphere2, sphere3, sphere4;
		//sphere1.center = Vec3f(0.812, 0.450, 1.722) + tr;
		sphere1.radius = 0.450;
		//sphere2.center = Vec3f(-0.812, 0.450, 1.722) + tr;
		sphere2.radius = 0.450;
		//sphere3.center = Vec3f(-0.812, 0.450, -1.426) + tr;
		sphere3.radius = 0.450;
		//sphere4.center = Vec3f(0.812, 0.450, -1.426) + tr;
		sphere4.radius = 0.450;

		//box5.center = Vec3f(0, 1.171, 1.722) + tr;
		box5.halfLength = Vec3f(0.8, 0.1, 0.1);

		RigidBodyInfo rigidbody;

		rigidbody.bodyId = i;

		rigidbody.position = Vec3f(0, 1.171, -0.011) + tr;
		auto bodyActor = jeep->addBox(box1, rigidbody, 100);

		box2.center = Vec3f(0.0f);
		rigidbody.position = Vec3f(0, 1.044, -2.254) + tr;
		auto spareTireActor = jeep->addBox(box2, rigidbody);

		Real wheel_velocity = 20;

		/*auto frontLeftTireActor = jeep->addCapsule(capsule1, rigidbody, 100);
		auto frontRightTireActor = jeep->addCapsule(capsule2, rigidbody, 100);
		auto rearLeftTireActor = jeep->addCapsule(capsule3, rigidbody, 100);
		auto rearRightTireActor = jeep->addCapsule(capsule4, rigidbody, 100);*/

		rigidbody.position = Vec3f(0.812, 0.450, 1.722) + tr;
		auto frontLeftTireActor = jeep->addSphere(sphere1, rigidbody, 50);

		rigidbody.position = Vec3f(-0.812, 0.450, 1.722) + tr;
		auto frontRightTireActor = jeep->addSphere(sphere2, rigidbody, 50);

		rigidbody.position = Vec3f(-0.812, 0.450, -1.426) + tr;
		auto rearLeftTireActor = jeep->addSphere(sphere3, rigidbody, 50);

		rigidbody.position = Vec3f(0.812, 0.450, -1.426) + tr;
		auto rearRightTireActor = jeep->addSphere(sphere4, rigidbody, 50);

		rigidbody.position = Vec3f(0, 1.171, 1.722) + tr;
		auto frontActor = jeep->addBox(box5, rigidbody, 25000);
		//auto rearActor = jeep->addBox(box6, rigidbody, 25000);

		//front rear
		auto& joint1 = jeep->createHingeJoint(frontLeftTireActor, frontActor);
		joint1.setAnchorPoint(frontLeftTireActor->center);
		//joint1.setMoter(wheel_velocity);
		joint1.setAxis(Vec3f(1, 0, 0));

		auto& joint2 = jeep->createHingeJoint(frontRightTireActor, frontActor);
		joint2.setAnchorPoint(frontRightTireActor->center);
		//joint2.setMoter(wheel_velocity);
		joint2.setAxis(Vec3f(1, 0, 0));

		//back rear
		auto& joint3 = jeep->createHingeJoint(rearLeftTireActor, bodyActor);
		joint3.setAnchorPoint(rearLeftTireActor->center);
		joint3.setMoter(wheel_velocity);
		joint3.setAxis(Vec3f(1, 0, 0));

		auto& joint4 = jeep->createHingeJoint(rearRightTireActor, bodyActor);
		joint4.setAnchorPoint(rearRightTireActor->center);
		joint4.setMoter(wheel_velocity);
		joint4.setAxis(Vec3f(1, 0, 0));


		auto& joint5 = jeep->createFixedJoint(bodyActor, spareTireActor);
		joint5.setAnchorPoint((bodyActor->center + spareTireActor->center) / 2);


		auto& joint6 = jeep->createHingeJoint(bodyActor, frontActor);
		joint6.setAnchorPoint(frontActor->center);
		joint6.setAxis(Vec3f(0, 1, 0));
		joint6.setRange(M_PI / 24, M_PI / 24);


		jeep->bind(bodyActor, Pair<uint, uint>(5, i));
		jeep->bind(spareTireActor, Pair<uint, uint>(4, i));
		jeep->bind(frontLeftTireActor, Pair<uint, uint>(0, i));
		jeep->bind(frontRightTireActor, Pair<uint, uint>(1, i));
		jeep->bind(rearLeftTireActor, Pair<uint, uint>(2, i));
		jeep->bind(rearRightTireActor, Pair<uint, uint>(3, i));
	}

	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());

	jeep->connect(multibody->importVehicles());

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varLocation()->setValue(Vec3f(0, 0, 0));
	plane->varScale()->setValue(Vec3f(300.0f));
	plane->stateTriangleSet()->connect(multibody->inTriangleSet());

	return scn;
}

int main()
{
	QtApp app;
	app.setSceneGraph(creatCar());
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter
	app.renderWindow()->getCamera()->setUnitScale(3.0f);

	app.mainLoop();

	return 0;
}


