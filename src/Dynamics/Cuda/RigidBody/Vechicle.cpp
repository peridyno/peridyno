#include "Vechicle.h"

#include "Module/SimpleVechicleDriver.h"
#include "Module/SharedFuncsForRigidBody.h"
#include "Module/ContactsUnion.h"
#include "Module/TJConstraintSolver.h"
#include "Module/TJSoftConstraintSolver.h"
#include "Module/PJSNJSConstraintSolver.h"
#include "Module/PJSoftConstraintSolver.h"
#include "Module/PJSConstraintSolver.h"

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"
#include "Collision/CollistionDetectionTriangleSet.h"
#include <GLWireframeVisualModule.h>
#include <Mapping/ContactsToEdgeSet.h>

#include "Module/CarDriver.h"

namespace dyno
{
	IMPLEMENT_TCLASS(Vechicle, TDataType)

	template<typename TDataType>
	Vechicle<TDataType>::Vechicle()
		: RigidBodySystem<TDataType>()
	{
		this->animationPipeline()->clear();

		auto elementQuery = std::make_shared<NeighborElementQuery<TDataType>>();
		elementQuery->varSelfCollision()->setValue(false);
		this->stateTopology()->connect(elementQuery->inDiscreteElements());
		this->stateCollisionMask()->connect(elementQuery->inCollisionMask());
		this->stateAttribute()->connect(elementQuery->inAttribute());
		this->animationPipeline()->pushModule(elementQuery);

		auto contactMapper = std::make_shared<ContactsToEdgeSet<DataType3f>>();
		elementQuery->outContacts()->connect(contactMapper->inContacts());
		contactMapper->varScale()->setValue(3.0);
		this->graphicsPipeline()->pushModule(contactMapper);

		auto wireRender = std::make_shared<GLWireframeVisualModule>();
		wireRender->setColor(Color(1, 0, 0));
		contactMapper->outEdgeSet()->connect(wireRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireRender);

// 		auto cdBV = std::make_shared<CollistionDetectionBoundingBox<TDataType>>();
// 		this->stateTopology()->connect(cdBV->inDiscreteElements());
// 		this->animationPipeline()->pushModule(cdBV);

		/*auto cdBV = std::make_shared<CollistionDetectionTriangleSet<TDataType>>();
		this->stateTopology()->connect(cdBV->inDiscreteElements());
		this->inTriangleSet()->connect(cdBV->inTriangleSet());*/
		auto cdBV = std::make_shared<CollistionDetectionBoundingBox<TDataType>>();
		this->stateTopology()->connect(cdBV->inDiscreteElements());
		this->animationPipeline()->pushModule(cdBV);


		auto merge = std::make_shared<ContactsUnion<TDataType>>();
		elementQuery->outContacts()->connect(merge->inContactsA());
		cdBV->outContacts()->connect(merge->inContactsB());
		this->animationPipeline()->pushModule(merge);

		auto iterSolver = std::make_shared<PJSConstraintSolver<TDataType>>();
		this->stateTimeStep()->connect(iterSolver->inTimeStep());
		this->varFrictionEnabled()->connect(iterSolver->varFrictionEnabled());
		this->varGravityEnabled()->connect(iterSolver->varGravityEnabled());
		this->varGravityValue()->connect(iterSolver->varGravityValue());
		//this->varFrictionCoefficient()->connect(iterSolver->varFrictionCoefficient());
		this->varFrictionCoefficient()->setValue(20.0f);
		this->varSlop()->connect(iterSolver->varSlop());
		this->stateMass()->connect(iterSolver->inMass());
		this->stateCenter()->connect(iterSolver->inCenter());
		this->stateVelocity()->connect(iterSolver->inVelocity());
		this->stateAngularVelocity()->connect(iterSolver->inAngularVelocity());
		this->stateRotationMatrix()->connect(iterSolver->inRotationMatrix());
		this->stateInertia()->connect(iterSolver->inInertia());
		this->stateQuaternion()->connect(iterSolver->inQuaternion());
		this->stateInitialInertia()->connect(iterSolver->inInitialInertia());

		this->stateTopology()->connect(iterSolver->inDiscreteElements());

		merge->outContacts()->connect(iterSolver->inContacts());

		this->animationPipeline()->pushModule(iterSolver);

		auto driver = std::make_shared<SimpleVechicleDriver>();

		this->stateFrameNumber()->connect(driver->inFrameNumber());
		this->stateInstanceTransform()->connect(driver->inInstanceTransform());

		this->animationPipeline()->pushModule(driver);

		this->inTriangleSet()->tagOptional(true);
	}

	template<typename TDataType>
	Vechicle<TDataType>::~Vechicle()
	{

	}

	template<typename TDataType>
	void Vechicle<TDataType>::resetStates()
	{
		RigidBodySystem<TDataType>::resetStates();

		auto topo = this->stateTopology()->constDataPtr();

		int sizeOfRigids = topo->totalSize();

		auto texMesh = this->inTextureMesh()->constDataPtr();

		uint N = texMesh->shapes().size();

		CArrayList<Transform3f> tms;
		CArray<uint> instanceNum(N);
		instanceNum.reset();

		//Calculate instance number
		for (uint i = 0; i < mBindingPair.size(); i++)
		{
			instanceNum[mBindingPair[i].first]++;
		}
		tms.resize(instanceNum);

		//Initialize CArrayList
		for (uint i = 0; i < N; i++)
		{
			for (uint j = 0; j < instanceNum[i]; j++)
			{
				tms[i].insert(Transform3f());
			}
		}

		this->stateInstanceTransform()->assign(tms);

		auto deTopo = this->stateTopology()->constDataPtr();
		auto offset = deTopo->calculateElementOffset();

		std::vector<Pair<uint, uint>> bindingPair(sizeOfRigids);
		std::vector<int> tags(sizeOfRigids, 0);

		for (int i = 0; i < mBindingPair.size(); i++)
		{
			auto actor = mActors[i];
			int idx = actor->idx + offset.checkElementOffset(actor->shapeType);

			bindingPair[idx] = mBindingPair[i];
			tags[idx] = 1;
		}

		mBindingPairDevice.assign(bindingPair);
		mBindingTagDevice.assign(tags);

		mInitialRot.assign(this->stateRotationMatrix()->constData());

		this->updateInstanceTransform();

		tms.clear();
		bindingPair.clear();
		tags.clear();
	}

	template<typename TDataType>
	void Vechicle<TDataType>::updateStates()
	{
		RigidBodySystem<TDataType>::updateStates();

		this->updateInstanceTransform();
	}

	template<typename TDataType>
	void Vechicle<TDataType>::updateInstanceTransform()
	{
		ApplyTransform(
			this->stateInstanceTransform()->getData(),
			this->stateOffset()->getData(),
			this->stateCenter()->getData(),
			this->stateRotationMatrix()->getData(),
			mInitialRot,
			mBindingPairDevice,
			mBindingTagDevice);
	}

	template<typename TDataType>
	void Vechicle<TDataType>::bind(std::shared_ptr<PdActor> actor, Pair<uint, uint> shapeId)
	{
		mActors.push_back(actor);
		mBindingPair.push_back(shapeId);
	}

	DEFINE_CLASS(Vechicle);

	//Jeep
	IMPLEMENT_TCLASS(Jeep, TDataType)

	template<typename TDataType>
	Jeep<TDataType>::Jeep()
		: ParametricModel<TDataType>()
		, Vechicle<TDataType>()
	{
		BoxInfo box1, box2, box3, box4;
		//car body
		box1.center = Vec3f(0, 1.171, -0.011);
		box1.halfLength = Vec3f(1.011, 0.793, 2.4);


		box2.center = Vec3f(0, 1.044, -2.254);
		box2.halfLength = Vec3f(0.447, 0.447, 0.15);

		box3.center = Vec3f(0.812, 0.450, 1.722);
		box3.halfLength = Vec3f(0.2f);
		box4.center = Vec3f(-0.812, 0.450, 1.722);
		box4.halfLength = Vec3f(0.2f);
// 		CapsuleInfo capsule1, capsule2, capsule3, capsule4;
// 
// 		capsule1.center = Vec3f(0.812, 0.450, 1.722);
// 		capsule1.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
// 		capsule1.halfLength = 0.1495;
// 		capsule1.radius = 0.450;
// 		capsule2.center = Vec3f(-0.812, 0.450, 1.722);
// 		capsule2.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
// 		capsule2.halfLength = 0.1495;
// 		capsule2.radius = 0.450;
// 		capsule3.center = Vec3f(-0.812, 0.450, -1.426);
// 		capsule3.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
// 		capsule3.halfLength = 0.1495;
// 		capsule3.radius = 0.450;
// 		capsule4.center = Vec3f(0.812, 0.450, -1.426);
// 		capsule4.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
// 		capsule4.halfLength = 0.1495;
// 		capsule4.radius = 0.450;

		SphereInfo sphere1, sphere2, sphere3, sphere4;
		sphere1.center = Vec3f(0.812, 0.450, 1.722);
		sphere1.radius = 0.450;
		sphere2.center = Vec3f(-0.812, 0.450, 1.722);
		sphere2.radius = 0.450;
		sphere3.center = Vec3f(-0.812, 0.450, -1.426);
		sphere3.radius = 0.450;
		sphere4.center = Vec3f(0.812, 0.450, -1.426);
		sphere4.radius = 0.450;

		RigidBodyInfo rigidbody;

		Vec3f offset = Vec3f(0.0f, -0.721f, 0.148f);
		rigidbody.offset = offset;
		auto bodyActor = this->addBox(box1, rigidbody, 10);

		rigidbody.offset = Vec3f(0.0f);

		auto spareTireActor = this->addBox(box2, rigidbody, 100);
		auto frontLeftSteerActor = this->addBox(box3, rigidbody, 1000);
		auto frontRightSteerActor = this->addBox(box4, rigidbody, 1000);

		Real wheel_velocity = 10;

// 		auto frontLeftTireActor = this->addCapsule(capsule1, rigidbody, 100);
// 		auto frontRightTireActor = this->addCapsule(capsule2, rigidbody, 100);
// 		auto rearLeftTireActor = this->addCapsule(capsule3, rigidbody, 100);
// 		auto rearRightTireActor = this->addCapsule(capsule4, rigidbody, 100);
		auto frontLeftTireActor = this->addSphere(sphere1, rigidbody, 100);
		auto frontRightTireActor = this->addSphere(sphere2, rigidbody, 100);
		auto rearLeftTireActor = this->addSphere(sphere3, rigidbody, 100);
		auto rearRightTireActor = this->addSphere(sphere4, rigidbody, 100);

		//front rear
		auto& joint1 = this->createHingeJoint(frontLeftTireActor, frontLeftSteerActor);
		joint1.setAnchorPoint(frontLeftTireActor->center);
		joint1.setMoter(wheel_velocity);
		joint1.setAxis(Vec3f(1, 0, 0));

		auto& joint2 = this->createHingeJoint(frontRightTireActor, frontRightSteerActor);
		joint2.setAnchorPoint(frontRightTireActor->center);
		joint2.setMoter(wheel_velocity);
		joint2.setAxis(Vec3f(1, 0, 0));

		//back rear
		auto& joint3 = this->createHingeJoint(rearLeftTireActor, bodyActor);
		joint3.setAnchorPoint(rearLeftTireActor->center);
		joint3.setMoter(wheel_velocity);
		joint3.setAxis(Vec3f(1, 0, 0));

		auto& joint4 = this->createHingeJoint(rearRightTireActor, bodyActor);
		joint4.setAnchorPoint(rearRightTireActor->center);
		joint4.setMoter(wheel_velocity);
		joint4.setAxis(Vec3f(1, 0, 0));


		//FixedJoint<Real> joint5(0, 1);
		auto& joint5 = this->createFixedJoint(bodyActor, spareTireActor);
		joint5.setAnchorPoint((bodyActor->center + spareTireActor->center) / 2);
		auto& joint6 = this->createFixedJoint(bodyActor, frontLeftSteerActor);
		joint6.setAnchorPoint((bodyActor->center + frontLeftSteerActor->center) / 2);
		auto& joint7 = this->createFixedJoint(bodyActor, frontRightSteerActor);
		joint7.setAnchorPoint((bodyActor->center + frontRightSteerActor->center) / 2);

		this->bind(bodyActor, Pair<uint, uint>(5, 0));
		this->bind(spareTireActor, Pair<uint, uint>(4, 0));
		this->bind(frontLeftTireActor, Pair<uint, uint>(0, 0));
		this->bind(frontRightTireActor, Pair<uint, uint>(1, 0));
		this->bind(rearLeftTireActor, Pair<uint, uint>(2, 0));
		this->bind(rearRightTireActor, Pair<uint, uint>(3, 0));


		auto driver = std::make_shared<CarDriver<DataType3f>>();
		this->animationPipeline()->pushModule(driver);
		this->stateQuaternion()->connect(driver->inQuaternion());
		this->stateTopology()->connect(driver->inTopology());
	}

	template<typename TDataType>
	Jeep<TDataType>::~Jeep()
	{

	}

	template<typename TDataType>
	void Jeep<TDataType>::resetStates()
	{
		Vechicle<TDataType>::resetStates();

		auto loc = this->varLocation()->getValue();
		
		Coord tr = Coord(loc.x, loc.y, loc.z);
		
		CArray<Coord> hostCenter;
		hostCenter.assign(this->stateCenter()->constData());

		for (uint i = 0; i < hostCenter.size(); i++)
		{
			hostCenter[i] += tr;
		}

		this->stateCenter()->assign(hostCenter);

		this->updateTopology();

		this->updateInstanceTransform();

		hostCenter.clear();
	}

	DEFINE_CLASS(Jeep);
}