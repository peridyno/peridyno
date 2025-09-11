#include "Vehicle.h"

#include "Module/CarDriver.h"
#include "Module/KeyDriver.h"

#include "Mapping/DiscreteElementsToTriangleSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

//Modeling
#include "GltfFunc.h"

//Rigidbody
#include "Module/InstanceTransform.h"

//Rendering
#include "Module/GLPhotorealisticInstanceRender.h"


namespace dyno
{
	//Jeep
	IMPLEMENT_TCLASS(Jeep, TDataType)

		template<typename TDataType>
	Jeep<TDataType>::Jeep() :
		ArticulatedBody<TDataType>()
	{
		auto driver = std::make_shared<CarDriver<DataType3f>>();
		this->stateTopology()->connect(driver->inTopology());
		this->animationPipeline()->pushModule(driver);
	}

	template<typename TDataType>
	Jeep<TDataType>::~Jeep()
	{

	}

	template<typename TDataType>
	void Jeep<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		std::string filename = getAssetPath() + "Jeep/JeepGltf/jeep.gltf";
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

			//wheel
			std::vector <int> Wheel_Id = { 0,1,2,3 };
			std::map<int, CapsuleInfo> wheels;
			std::map<int, std::shared_ptr<PdActor>> actors;
			//Capsule
			for (auto it : Wheel_Id)
			{
				auto up = texMesh->shapes()[it]->boundingBox.v1;
				auto down = texMesh->shapes()[it]->boundingBox.v0;

				wheels[it].center = Vec3f(0.0f);
				wheels[it].rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
				wheels[it].halfLength = 0.1;
				wheels[it].radius = std::abs(up.y - down.y) / 2;

				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				actors[it] = this->addCapsule(wheels[it], rigidbody, 100);
			}


			//body
			int body = 5;
			int backWheel = 4;

			std::vector <int> box_Id = { body,backWheel };
			std::map<int, BoxInfo> boxs;
			//box
			for (auto it : box_Id)
			{
				auto up = texMesh->shapes()[it]->boundingBox.v1;
				auto down = texMesh->shapes()[it]->boundingBox.v0;

				boxs[it].center = Vec3f(0.0f);

				boxs[it].halfLength = (up - down) / 2;

				rigidbody.offset = Vec3f(0.0f, 0.0f, 0.0f);
				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());
				actors[it] = this->addBox(boxs[it], rigidbody, 100);
			}

			//bindShapetoActor
			for (auto it : box_Id)
			{
				this->bindShape(actors[it], Pair<uint, uint>(it, i));
			}
			//bindShapetoActor
			for (auto it : Wheel_Id)
			{
				this->bindShape(actors[it], Pair<uint, uint>(it, i));
			}

			rigidbody.offset = Vec3f(0);

			Real wheel_velocity = 10;

			//wheel to Body
			for (auto it : Wheel_Id)
			{
				auto& joint = this->createHingeJoint(actors[it], actors[body]);
				joint.setAnchorPoint(actors[it]->center);
				joint.setMoter(wheel_velocity);
				joint.setAxis(Quat1f(instances[i].rotation()).rotate(Vec3f(1, 0, 0)));
			}

			auto& jointBackWheel_Body = this->createFixedJoint(actors[backWheel], actors[body]);
			jointBackWheel_Body.setAnchorPoint(actors[backWheel]->center);		//set and offset
		}

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
	}

	DEFINE_CLASS(Jeep);

	//Tank
	IMPLEMENT_TCLASS(Tank, TDataType)

		template<typename TDataType>
	Tank<TDataType>::Tank() :
		ArticulatedBody<TDataType>()
	{
		auto driver = std::make_shared<CarDriver<DataType3f>>();
		this->stateTopology()->connect(driver->inTopology());
		this->animationPipeline()->pushModule(driver);

		auto instance = std::make_shared<InstanceTransform<DataType3f>>();

		this->stateCenter()->connect(instance->inCenter());
		this->stateRotationMatrix()->connect(instance->inRotationMatrix());
		this->stateBindingPair()->connect(instance->inBindingPair());
		this->stateBindingTag()->connect(instance->inBindingTag());

		this->animationPipeline()->pushModule(instance);
	}

	template<typename TDataType>
	Tank<TDataType>::~Tank()
	{

	}

	template<typename TDataType>
	void Tank<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		std::string filename = getAssetPath() + "gltf/Tank/Tank.gltf";
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

			//wheel
			std::vector <int> Wheel_Id = { 2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
			std::map<int, CapsuleInfo> wheels;
			std::map<int, std::shared_ptr<PdActor>> actors;
			//Capsule
			for (auto it : Wheel_Id)
			{
				auto up = texMesh->shapes()[it]->boundingBox.v1;
				auto down = texMesh->shapes()[it]->boundingBox.v0;

				wheels[it].center = Vec3f(0.0f);
				wheels[it].rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
				wheels[it].halfLength = 0.1;
				wheels[it].radius = std::abs(up.y - down.y) / 2;

				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());
				actors[it] = this->addCapsule(wheels[it], rigidbody, 100);
			}


			//body
			int body = 0;
			int gun = 1;

			std::vector <int> box_Id = { body,gun };
			std::map<int, BoxInfo> boxs;
			//box
			for (auto it : box_Id)
			{
				auto up = texMesh->shapes()[it]->boundingBox.v1;
				auto down = texMesh->shapes()[it]->boundingBox.v0;

				boxs[it].center = Vec3f(0.0f);

				boxs[it].halfLength = (up - down) / 2 * 0.5;

				rigidbody.offset = Vec3f(0.0f, 0.0f, 0.0f);
				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());
				actors[it] = this->addBox(boxs[it], rigidbody, 100);
			}

			//bindShapetoActor
			for (auto it : box_Id)
			{
				this->bindShape(actors[it], Pair<uint, uint>(it, i));
			}
			//bindShapetoActor
			for (auto it : Wheel_Id)
			{
				this->bindShape(actors[it], Pair<uint, uint>(it, i));
			}

			rigidbody.offset = Vec3f(0);

			Real wheel_velocity = 10;

			//wheel to Body
			for (auto it : Wheel_Id)
			{
				auto& joint = this->createHingeJoint(actors[it], actors[body]);
				joint.setAnchorPoint(actors[it]->center);
				joint.setMoter(wheel_velocity);
				joint.setAxis(Quat1f(instances[i].rotation()).rotate(Vec3f(1, 0, 0)));
			}

			auto& jointGun_Body = this->createFixedJoint(actors[gun], actors[body]);
			jointGun_Body.setAnchorPoint(actors[gun]->center);		//set and offset
		}

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
	}

	DEFINE_CLASS(Tank);

	//TrackedTank
	IMPLEMENT_TCLASS(TrackedTank, TDataType)

		template<typename TDataType>
	TrackedTank<TDataType>::TrackedTank() :
		ArticulatedBody<TDataType>()
	{
		auto driver = std::make_shared<CarDriver<DataType3f>>();
		this->stateTopology()->connect(driver->inTopology());
		this->animationPipeline()->pushModule(driver);

		this->statecaterpillarTrack()->setDataPtr(std::make_shared<EdgeSet<DataType3f>>());
		auto wireframe = std::make_shared<GLWireframeVisualModule>();
		this->statecaterpillarTrack()->connect(wireframe->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireframe);

	}

	template<typename TDataType>
	TrackedTank<TDataType>::~TrackedTank()
	{

	}


	template<typename TDataType>
	void TrackedTank<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		std::string filename = getAssetPath() + "gltf/Tank/TrackedTank.gltf";
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
			std::map<int, std::shared_ptr<PdActor>> actors;

#define FULLBODY

#ifdef FULLBODY
			//wheel
			std::vector <int> Wheel_Id = { 1,2,3,4,6,7,8,10,11,12,13,15,16,17 };
			//Capsule
			for (auto it : Wheel_Id)
			{
				auto up = texMesh->shapes()[it]->boundingBox.v1;
				auto down = texMesh->shapes()[it]->boundingBox.v0;

				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());
				rigidbody.motionType = BodyType::Dynamic;
				auto actor = this->createRigidBody(rigidbody);
				actors[it] = actor;

				CapsuleInfo capsule;
				capsule.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
				capsule.radius = (up[1] - down[1]) / 2;
				capsule.halfLength = (up[2] - down[2]) / 2;

				float r = (up[1] - down[1]) / 2;
				this->bindCapsule(actor, capsule, 1000);
				this->bindShape(actor, Pair<uint, uint>(it, i));

			}


			//Gear
			std::vector<int> gear = { 0,5,9,14 };
			for (size_t c = 0; c < 4; c++)
			{
				int cid = gear[c];
				auto up = texMesh->shapes()[cid]->boundingBox.v1;
				auto down = texMesh->shapes()[cid]->boundingBox.v0;
				//first gear
				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[cid]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				rigidbody.motionType = BodyType::Dynamic;
				auto actor = this->createRigidBody(rigidbody);
				actors[cid] = actor;

				for (uint sec = 0; sec < 14; sec++)
				{
					CapsuleInfo capsule;
					capsule.radius = 0.015;
					capsule.halfLength = (up[1] - down[1]) / 2;
					Vec3f offset = sec <= 6 ? Vec3f(-0.1, 0, 0) : Vec3f(0.1, 0, 0);
					float theta = sec * M_PI / 7;

					capsule.rot = Quat1f(sec * M_PI / 7 + M_PI / 14, Vec3f(1, 0, 0));
					capsule.center = offset;
					this->bindCapsule(actor, capsule, 100000);

					//row
					float r = (up[1] - down[1]) / 3;
					float theta2 = sec * M_PI / 7;
					float y = r * sin(theta2);
					float z = r * cos(theta2);

					capsule.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
					capsule.radius = 0.05f;
					capsule.halfLength = (up[0] - down[0]) / 2;
					capsule.center = Vec3f(0, y, z);
					this->bindCapsule(actor, capsule, 100000);
				}

				this->bindShape(actor, Pair<uint, uint>(cid, i));


			}

			//Gun
			int head = 18;
			{
				auto up = texMesh->shapes()[head]->boundingBox.v1;
				auto down = texMesh->shapes()[head]->boundingBox.v0;

				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[head]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				rigidbody.motionType = BodyType::Dynamic;
				auto actor = this->createRigidBody(rigidbody);
				actors[head] = actor;

				BoxInfo box;
				box.halfLength = Vec3f(0.8, 0.4, 1.2);
				box.center = Vec3f(0, -0.2, -1);

				this->bindBox(actor, box);

				CapsuleInfo capsule;
				capsule.rot = Quat1f(M_PI / 2, Vec3f(1, 0, 0));
				capsule.radius = 0.1f;
				capsule.halfLength = (up[2] - down[2]) / 3;
				capsule.center = Vec3f(0.3, -0.2, 0.5);
				this->bindCapsule(actor, capsule);
				capsule.center = Vec3f(-0.3, -0.2, 0.5);
				this->bindCapsule(actor, capsule);

				this->bindShape(actor, Pair<uint, uint>(head, i));
			}
			//Body
			int body = 19;
			{
				auto up = texMesh->shapes()[body]->boundingBox.v1;
				auto down = texMesh->shapes()[body]->boundingBox.v0;

				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[body]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				rigidbody.motionType = BodyType::Dynamic;
				auto actor = this->createRigidBody(rigidbody);
				actors[body] = actor;

				BoxInfo box;
				box.rot = Quat1f(0, Vec3f(0, 0, 1));
				box.halfLength = (up - down) / Vec3f(3, 2, 3);

				this->bindBox(actor, box);
				this->bindShape(actor, Pair<uint, uint>(body, i));
			}
#endif // FULLBODY

			std::vector<int> caterpillarTrack_L;
			std::vector<int> caterpillarTrack_R;

			for (int cid = 20; cid < 160; cid++)
			{
				auto up = texMesh->shapes()[cid]->boundingBox.v1;
				auto down = texMesh->shapes()[cid]->boundingBox.v0;

				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[cid]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				rigidbody.motionType = BodyType::Dynamic;
				rigidbody.bodyId = i * 2 + 1;
				auto actor = this->createRigidBody(rigidbody);
				actors[cid] = actor;
				if (cid < 90)
					caterpillarTrack_L.push_back(cid);
				else
					caterpillarTrack_R.push_back(cid);


				SphereInfo capsule;
				capsule.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
				capsule.radius = 0.05;
				if(cid < 90)
					capsule.center = Vec3f((up[0] - down[0]) / 2 - 0.05, 0, 0);
				else
					capsule.center = Vec3f((up[0] - down[0]) / 2 + 0.05, 0, 0);
				//capsule.halfLength = (up[0] - down[0]) / 2;

				this->bindSphere(actor, capsule, 1000000);
				capsule.center = Vec3f(-(up[0] - down[0]) / 2, 0, 0);
				this->bindSphere(actor, capsule, 1000000);

				this->bindShape(actor, Pair<uint, uint>(cid, i));
			}

#ifdef FULLBODY
			Real wheel_velocity = 10;

			//weel to Body Joint
			for (auto it : Wheel_Id)
			{
				auto& joint = this->createHingeJoint(actors[it], actors[body]);
				joint.setAnchorPoint(actors[it]->center);
				//joint.setMoter(wheel_velocity);
				joint.setAxis(Quat1f(instances[i].rotation()).rotate(Vec3f(1, 0, 0)));
			}

			//Gear to Body Joint
			for (auto it : gear)
			{
				auto& joint = this->createHingeJoint(actors[it], actors[body]);
				joint.setAnchorPoint(actors[it]->center);
				joint.setMoter(wheel_velocity);
				joint.setAxis(Quat1f(instances[i].rotation()).rotate(Vec3f(1, 0, 0)));
			}

			auto& jointGun_Body = this->createFixedJoint(actors[head], actors[body]);
			jointGun_Body.setAnchorPoint(actors[body]->center);

#endif // FULLBODY

			//Caterpillar Track Joint
			auto edgeset = this->statecaterpillarTrack()->getDataPtr();
			std::vector<TopologyModule::Edge> edges;
			std::vector<Vec3f> points;
			for (int k = 0; k < caterpillarTrack_L.size(); k++)
			{
				auto start = caterpillarTrack_L[k];
				auto end = k != caterpillarTrack_L.size() - 1 ? caterpillarTrack_L[k + 1] : caterpillarTrack_L[0];

				auto& joint = this->createHingeJoint(actors[start], actors[end]);
				joint.setAnchorPoint((actors[start]->center + actors[end]->center) / 2);
				joint.setAxis(Quat1f(instances[i].rotation()).rotate(Vec3f(1, 0, 0)));

				points.push_back(actors[start]->center);

				if (k != caterpillarTrack_L.size() - 1)
					edges.push_back(TopologyModule::Edge(k, k + 1));
				else
					edges.push_back(TopologyModule::Edge(k, 0));


			}
			auto edgeOffset = points.size();
			for (int k = 0; k < caterpillarTrack_R.size(); k++)
			{
				auto start = caterpillarTrack_R[k];
				auto end = k != caterpillarTrack_R.size() - 1 ? caterpillarTrack_R[k + 1] : caterpillarTrack_R[0];

				auto& joint = this->createHingeJoint(actors[start], actors[end]);
				joint.setAnchorPoint((actors[start]->center + actors[end]->center) / 2);
				joint.setAxis(Quat1f(instances[i].rotation()).rotate(Vec3f(1, 0, 0)));

				points.push_back(actors[start]->center);

				if (k != caterpillarTrack_R.size() - 1)
					edges.push_back(TopologyModule::Edge(k + edgeOffset, k + 1 + edgeOffset));
				else
					edges.push_back(TopologyModule::Edge(k + edgeOffset, 0 + edgeOffset));

			}
			edgeset->setPoints(points);
			edgeset->setEdges(edges);
			edgeset->update();


		}

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();

	}

	DEFINE_CLASS(TrackedTank);

	//UAV
	IMPLEMENT_TCLASS(UAV, TDataType)

		template<typename TDataType>
	UAV<TDataType>::UAV() :
		ArticulatedBody<TDataType>()
	{
		auto driver = std::make_shared<CarDriver<DataType3f>>();
		this->stateTopology()->connect(driver->inTopology());
		this->animationPipeline()->pushModule(driver);
	}

	template<typename TDataType>
	UAV<TDataType>::~UAV()
	{

	}

	template<typename TDataType>
	void UAV<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		std::string filename = getAssetPath() + "gltf/UAV/UAV.gltf";
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

			//wheel
			std::vector <int> Wheel_Id = { 0,1,2,3 };
			std::map<int, CapsuleInfo> wheels;
			std::map<int, std::shared_ptr<PdActor>> actors;
			//Capsule
			for (auto it : Wheel_Id)
			{
				auto up = texMesh->shapes()[it]->boundingBox.v1;
				auto down = texMesh->shapes()[it]->boundingBox.v0;

				wheels[it].center = Vec3f(0.0f);

				auto rot = it == 0 || it == 3 ? Vec3f(90, 45, 0) : Vec3f(90, -45, 0);

				Quat<Real> q =
					Quat<Real>(Real(M_PI) * rot[2] / 180, Coord(0, 0, 1))
					* Quat<Real>(Real(M_PI) * rot[1] / 180, Coord(0, 1, 0))
					* Quat<Real>(Real(M_PI) * rot[0] / 180, Coord(1, 0, 0));
				q.normalize();


				wheels[it].rot = q;
				wheels[it].halfLength = std::abs(up.x - down.x) / 1.5;
				wheels[it].radius = std::abs(up.y - down.y) / 2;

				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				actors[it] = this->addCapsule(wheels[it], rigidbody, 100);
			}


			//body
			int body = 4;

			std::vector <int> box_Id = { body };
			std::map<int, BoxInfo> boxs;
			//box
			for (auto it : box_Id)
			{
				auto up = texMesh->shapes()[it]->boundingBox.v1;
				auto down = texMesh->shapes()[it]->boundingBox.v0;

				boxs[it].center = Vec3f(0.0f);

				boxs[it].halfLength = (up - down) / 2;

				rigidbody.offset = Vec3f(0.0f, 0.0f, 0.0f);
				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				actors[it] = this->addBox(boxs[it], rigidbody, 100);
			}

			//bindShapetoActor
			for (auto it : box_Id)
			{
				this->bindShape(actors[it], Pair<uint, uint>(it, i));
			}
			//bindShapetoActor
			for (auto it : Wheel_Id)
			{
				this->bindShape(actors[it], Pair<uint, uint>(it, i));
			}

			rigidbody.offset = Vec3f(0);



			//wheel to Body
			for (auto it : Wheel_Id)
			{
				Real wheel_velocity = (it == 0 || it == 3) ? Real(20.0) : Real(-20.0);

				auto& joint = this->createHingeJoint(actors[it], actors[body]);
				joint.setAnchorPoint(actors[it]->center);
				joint.setMoter(wheel_velocity);
				joint.setAxis(Quat1f(instances[i].rotation()).rotate(Vec3f(0, 1, 0)));
			}

		}

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
	}

	DEFINE_CLASS(UAV);


	//UUV
	IMPLEMENT_TCLASS(UUV, TDataType)

		template<typename TDataType>
	UUV<TDataType>::UUV() :
		ArticulatedBody<TDataType>()
	{
		auto driver = std::make_shared<CarDriver<DataType3f>>();
		this->stateTopology()->connect(driver->inTopology());
		this->animationPipeline()->pushModule(driver);
	}

	template<typename TDataType>
	UUV<TDataType>::~UUV()
	{

	}

	template<typename TDataType>
	void UUV<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		std::string filename = getAssetPath() + "gltf/UUV/UUV.gltf";
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

			//wheel
			std::vector <int> Wheel_Id = { 1,2,3,4 };
			std::map<int, CapsuleInfo> wheels;
			std::map<int, std::shared_ptr<PdActor>> actors;
			//Capsule
			for (auto it : Wheel_Id)
			{
				auto up = texMesh->shapes()[it]->boundingBox.v1;
				auto down = texMesh->shapes()[it]->boundingBox.v0;

				wheels[it].center = Vec3f(0.0f);

				auto rot = it == 1 || it == 2 ? Vec3f(90, 30, 0) : Vec3f(0, 0, 30);

				Quat<Real> q =
					Quat<Real>(Real(M_PI) * rot[2] / 180, Coord(0, 0, 1))
					* Quat<Real>(Real(M_PI) * rot[1] / 180, Coord(0, 1, 0))
					* Quat<Real>(Real(M_PI) * rot[0] / 180, Coord(1, 0, 0));
				q.normalize();


				wheels[it].rot = q;
				wheels[it].halfLength = 0.13;
				wheels[it].radius = 0.03;

				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				actors[it] = this->addCapsule(wheels[it], rigidbody, 100);
			}


			//body
			int body = 0;

			std::vector <int> box_Id = { body };
			std::map<int, BoxInfo> boxs;
			//box
			for (auto it : box_Id)
			{
				auto up = texMesh->shapes()[it]->boundingBox.v1;
				auto down = texMesh->shapes()[it]->boundingBox.v0;

				boxs[it].center = Vec3f(0.0f);

				boxs[it].halfLength = (up - down) / 2;

				rigidbody.offset = Vec3f(0.0f, 0.0f, 0.0f);
				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				actors[it] = this->addBox(boxs[it], rigidbody, 100);
			}

			//bindShapetoActor
			for (auto it : box_Id)
			{
				this->bindShape(actors[it], Pair<uint, uint>(it, i));
			}
			//bindShapetoActor
			for (auto it : Wheel_Id)
			{
				this->bindShape(actors[it], Pair<uint, uint>(it, i));
			}

			rigidbody.offset = Vec3f(0);



			//wheel to Body
			for (auto it : Wheel_Id)
			{
				Real wheel_velocity = 20;

				auto& joint = this->createHingeJoint(actors[it], actors[body]);
				joint.setAnchorPoint(actors[it]->center);
				joint.setMoter(wheel_velocity);
				Vec3f axis = it == 1 || it == 2 ? Vec3f(0, 1, 0) : Vec3f(0, 0, 1);
				joint.setAxis(Quat1f(instances[i].rotation()).rotate(axis));
			}

		}

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
	}

	DEFINE_CLASS(UUV);



	//Bicycle
	IMPLEMENT_TCLASS(Bicycle, TDataType)

	template<typename TDataType>
	Bicycle<TDataType>::Bicycle() :
		ArticulatedBody<TDataType>()
	{
		
		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		this->stateTopology()->connect(mapper->inDiscreteElements());
		this->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->setColor(Color(1, 1, 0));
		sRender->setAlpha(0.5);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);

		
	}

	template<typename TDataType>
	Bicycle<TDataType>::~Bicycle()
	{

	}

	template<typename TDataType>
	void Bicycle<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		std::string filename = getAssetPath() + "gltf/bicycle/bicycle.gltf";
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
			std::map<int, std::shared_ptr<PdActor>> actors;

			//wheel
			std::vector <int> Wheel_Id = { 2,3};
			//Capsule
			for (auto it : Wheel_Id)
			{
				auto up = texMesh->shapes()[it]->boundingBox.v1;
				auto down = texMesh->shapes()[it]->boundingBox.v0;

				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());
				rigidbody.motionType = BodyType::Dynamic;
				auto actor = this->createRigidBody(rigidbody);
				actors[it] = actor;

				int sample = 80;

//#define CAPSULEMODE
#define MULTISPHEREMODE

#ifdef CAPSULEMODE
				for (size_t j = 0; j < sample; j++)
				{
					float capsuleAngle = 360.0f / float(sample) * float(j);
					CapsuleInfo capsule;
					capsule.rot = Quat1f(glm::radians(capsuleAngle), Vec3f(0, 0, 1));
					capsule.radius = abs(up[2] - down[2]) / 4;
					capsule.halfLength = abs(up[1] - down[1]) / 2 - capsule.radius;

					float r = (up[1] - down[1]) / 2;
					this->bindCapsule(actor, capsule, 1000);
				}


#endif //CAPSULEMODE

#ifdef MULTISPHEREMODE			
				for (size_t j = 0; j < sample; j++)
				{
					float capsuleAngle = 360.0f / float(sample) * float(j);
					SphereInfo sphere;
					sphere.radius = abs(up[2] - down[2]) / 4;
					Quat1f q = Quat1f(glm::radians(capsuleAngle), Vec3f(0, 0, 1));
					float pr = abs(up[1] - down[1]) / 2;
					sphere.center = q.rotate(Vec3f(0,pr - sphere.radius, 0 ));

					float r = (up[1] - down[1]) / 2;
					this->bindSphere(actor, sphere, 1000);
				}
#endif //MULTISPHEREMODE

				this->bindShape(actor, Pair<uint, uint>(it, i));

			}

			std::vector <int> bodyElement_Id = { 0,1,4,5 };

			//Body
			for (auto it : bodyElement_Id)
			{			
				auto up = texMesh->shapes()[it]->boundingBox.v1;
				auto down = texMesh->shapes()[it]->boundingBox.v0;

				rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation()) + instances[i].translation();
				rigidbody.angle = Quat1f(instances[i].rotation());

				rigidbody.motionType = BodyType::Dynamic;
				auto actor = this->createRigidBody(rigidbody);
				actors[it] = actor;

				BoxInfo box;
				box.rot = Quat1f(0, Vec3f(0, 0, 1));
				box.halfLength = (up - down) / 2;

				this->bindBox(actor, box);		
				this->bindShape(actor, Pair<uint, uint>(it, i));

			}

			Real wheel_velocity = 5;

			int main = 1;
			int frontwheel = 2;
			int backwheel = 3;
			int handle = 4;
			int ext1 = 0;
			int ext2 = 5;

			auto& frontWheelJoint = this->createHingeJoint(actors[frontwheel], actors[handle]);
			frontWheelJoint.setAnchorPoint(actors[frontwheel]->center);
			frontWheelJoint.setMoter(wheel_velocity);
			frontWheelJoint.setAxis(Vec3f(0, 0, -1));

			auto& backWheelJoint = this->createHingeJoint(actors[backwheel], actors[main]);
			backWheelJoint.setAnchorPoint(actors[backwheel]->center);
			backWheelJoint.setMoter(wheel_velocity);
			backWheelJoint.setAxis(Vec3f(0, 0, -1));

			auto& handleJoint = this->createHingeJoint(actors[handle], actors[main]);
			handleJoint.setAnchorPoint(actors[handle]->center);
			handleJoint.setRange(-M_PI * 2 / 3, M_PI * 2 / 3);
			handleJoint.setAxis(Quat1f(glm::radians(-21.833), Vec3f(0, 0, 1)).rotate(Vec3f(0, 1, 0)));

			auto& extFix1 = this->createFixedJoint(actors[ext1], actors[main]);
			extFix1.setAnchorPoint(actors[main]->center);

			auto& extFix2 = this->createFixedJoint(actors[ext2], actors[handle]);
			extFix2.setAnchorPoint(actors[main]->center);
		}

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
		this->outReset()->setValue(true);
	}

	DEFINE_CLASS(Bicycle);
}
