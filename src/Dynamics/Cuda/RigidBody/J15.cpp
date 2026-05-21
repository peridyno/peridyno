#include "J15.h"

#include "RigidBody/Module/CarDriver.h"

#include "Mapping/DiscreteElementsToTriangleSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

//Modeling
#include "GltfFunc.h"

//Rigidbody
#include "RigidBody/Module/InstanceTransform.h"

//Rendering
#include "Module/GLPhotorealisticInstanceRender.h"


namespace dyno
{

	//J15
	IMPLEMENT_TCLASS(J15, TDataType)

	template<typename TDataType>
	J15<TDataType>::J15() :
		ArticulatedBody<TDataType>()
	{
		auto driver = std::make_shared<CarDriver<DataType3f>>();
		this->stateTopology()->connect(driver->inTopology());
		this->animationPipeline()->pushModule(driver);

		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		this->stateTopology()->connect(mapper->inDiscreteElements());
		this->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->varBaseColor()->setValue(Color(1, 1, 0));
		sRender->varAlpha()->setValue(0.15);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);
	}

	template<typename TDataType>
	J15<TDataType>::~J15()
	{

	}

	template<typename TDataType>
	void J15<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		std::string filename = getAssetPath() + "J15/J15.obj";
		if (this->varFilePath()->getValue() != filename)
		{
			this->varFilePath()->setValue(FilePath(filename));
		}

		auto instances = this->varVehiclesTransform()->getValue();
		uint vehicleNum = instances.size();
		for (size_t i = 0; i < vehicleNum; i++)
		{
			RigidBodyInfo rigidbody;
			rigidbody.bodyId = i;
			rigidbody.friction = this->varFrictionCoefficient()->getValue();

			auto texMesh = this->stateTextureMesh()->constDataPtr();

			std::vector<int> wheels = { 1,2,7 };
			std::vector<std::shared_ptr<dyno::PdActor>> wheelActors;

			for (auto id : wheels)
			{
				auto rot =  Vec3f(90, 0, 0);
				Quat<Real> q =
					Quat<Real>(Real(M_PI) * rot[2] / 180, Coord(0, 0, 1))
					* Quat<Real>(Real(M_PI) * rot[1] / 180, Coord(0, 1, 0))
					* Quat<Real>(Real(M_PI) * rot[0] / 180, Coord(1, 0, 0));
				q.normalize();

				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[id]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				std::shared_ptr<dyno::PdActor> wheelActor = this->createRigidBody(rigidbody);
				wheelActors.push_back(wheelActor);

				CapsuleInfo capsule;
				capsule.center = Vec3f(0.0f);
				capsule.rot = q;
				auto up = texMesh->shapes()[id]->boundingBox.v1;
				auto down = texMesh->shapes()[id]->boundingBox.v0;
				capsule.radius = std::abs(up.y - down.y) / 2;
				capsule.halfLength = std::abs(up.x - down.x) / 1.5;
				this->bindCapsule(wheelActor, capsule);

				this->bindShape(wheelActor, Pair<uint, uint>(id, i));
			}

			std::vector<int> undercarts = { 4,3,6 };
			std::vector<std::shared_ptr<dyno::PdActor>> undercartActors;
			for (auto id : undercarts)
			{
				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[id]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				std::shared_ptr<dyno::PdActor> undercartActor = this->createRigidBody(rigidbody);
				undercartActors.push_back(undercartActor);

				CapsuleInfo capsule;
				capsule.center = Vec3f(0.0f);
				auto up = texMesh->shapes()[id]->boundingBox.v1;
				auto down = texMesh->shapes()[id]->boundingBox.v0;
				capsule.radius = std::abs(up.z - down.z) / 2;
				capsule.halfLength = std::abs(up.y - down.y) / 2.2;
				this->bindCapsule(undercartActor, capsule);

				this->bindShape(undercartActor, Pair<uint, uint>(id, i));
			}

			int bodyId = 5;
			std::shared_ptr<dyno::PdActor> bodyActor;
			{
				auto rot = Vec3f(0.0f,0.0f, 90.0f);
				Quat<Real> q =
					Quat<Real>(Real(M_PI) * rot[2] / 180, Coord(0, 0, 1))
					* Quat<Real>(Real(M_PI) * rot[1] / 180, Coord(0, 1, 0))
					* Quat<Real>(Real(M_PI) * rot[0] / 180, Coord(1, 0, 0));
				q.normalize();

				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[bodyId]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				bodyActor = this->createRigidBody(rigidbody);
				CapsuleInfo capsule;
				capsule.center = Vec3f(0.0f, - 0.50f,0.0f);
				capsule.rot = q;
				capsule.radius = 0.5;
				capsule.halfLength = 19.3/2 - 0.6;
				this->bindCapsule(bodyActor, capsule);

				rot = Vec3f(23.061, 0, -90.0f);
				q =
					Quat<Real>(Real(M_PI) * rot[2] / 180, Coord(0, 0, 1))
					* Quat<Real>(Real(M_PI) * rot[1] / 180, Coord(0, 1, 0))
					* Quat<Real>(Real(M_PI) * rot[0] / 180, Coord(1, 0, 0));
				q.normalize();

				capsule.center = Vec3f(1.513f, -0.50f,-3.295f);
				capsule.rot = q;
				capsule.radius = 0.5;
				capsule.halfLength = 19.3/2 - 0.6;
				this->bindCapsule(bodyActor, capsule);

				rot = Vec3f(-23.061, 0, -90.0f);
				q =
					Quat<Real>(Real(M_PI) * rot[2] / 180, Coord(0, 0, 1))
					* Quat<Real>(Real(M_PI) * rot[1] / 180, Coord(0, 1, 0))
					* Quat<Real>(Real(M_PI) * rot[0] / 180, Coord(1, 0, 0));
				q.normalize();

				capsule.center = Vec3f(1.513f, -0.50f, 3.295f);
				capsule.rot = q;
				capsule.radius = 0.5;
				capsule.halfLength = 19.3 / 2 - 0.6;
				this->bindCapsule(bodyActor, capsule);

				rot = Vec3f(-90.0f, 0.0f, -90.0f);
				q =
					Quat<Real>(Real(M_PI) * rot[2] / 180, Coord(0, 0, 1))
					* Quat<Real>(Real(M_PI) * rot[1] / 180, Coord(0, 1, 0))
					* Quat<Real>(Real(M_PI) * rot[0] / 180, Coord(1, 0, 0));
				q.normalize();

				capsule.center = Vec3f(-9.032f, -0.719f, 0.0f);
				capsule.rot = q;
				capsule.radius = 0.4;
				capsule.halfLength = 7.7 / 2;
				this->bindCapsule(bodyActor, capsule);

				BoxInfo box;
				box.center = Vec3f(-7.458f, 1.453f, 0.0f);
				box.halfLength = Vec3f(0.5f, 1.2f, 2.05f);

				rot = Vec3f(0.0f, 6.193f, 0.0f);
				q =
					Quat<Real>(Real(M_PI) * rot[2] / 180, Coord(0, 0, 1))
					* Quat<Real>(Real(M_PI) * rot[1] / 180, Coord(0, 1, 0))
					* Quat<Real>(Real(M_PI) * rot[0] / 180, Coord(1, 0, 0));
				q.normalize();
				this->bindBox(bodyActor, box);
				this->bindShape(bodyActor, Pair<uint, uint>(bodyId, i));
			}

			int fCover = 0;
			std::shared_ptr<dyno::PdActor> fCoverActor;
			{
				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[fCover]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				fCoverActor = this->createRigidBody(rigidbody);

				auto up = texMesh->shapes()[fCover]->boundingBox.v1;
				auto down = texMesh->shapes()[fCover]->boundingBox.v0;

				BoxInfo box;
				box.halfLength = Vec3f(std::abs(up.x - down.x) / 2, std::abs(up.y - down.y) / 2, std::abs(up.z - down.z) / 2);

				this->bindBox(fCoverActor, box);
				this->bindShape(fCoverActor, Pair<uint, uint>(fCover, i));
			}

			for (size_t wid = 0; wid < wheelActors.size(); wid++)
			{
				auto wheelActor = wheelActors[wid];
				auto undercartActor = undercartActors[wid];

				auto& wheelJoint = this->createHingeJoint(wheelActor, undercartActor);
				wheelJoint.setAnchorPoint(wheelActor->center);
				wheelJoint.setAxis(Vec3f(0, 0, 1));
				wheelJoint.setMoter(-10);

				auto& undercartJoint = this->createHingeJoint(undercartActor, bodyActor);
				undercartJoint.setAnchorPoint(undercartActor->center + 0.9f);
				undercartJoint.setAxis(Vec3f(0, 0, 1));
				undercartJoint.setRange(0.0f, 0.000001f);
			}

			auto& coverJoint = this->createHingeJoint(fCoverActor, bodyActor);
			coverJoint.setAnchorPoint(fCoverActor->center + 0.35f);
			coverJoint.setAxis(Vec3f(0, 0, 1));
			coverJoint.setRange(0.0f, 0.000001f);

		}

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
	}

	DEFINE_CLASS(J15);
}
