#include "PresetArticulatedBody.h"

#include "Module/CarDriver.h"

#include "Mapping/DiscreteElementsToTriangleSet.h"
#include "GLSurfaceVisualModule.h"

//Modeling
#include "GltfFunc.h"
//Rigidbody
#include "Module/InstanceTransform.h"
//Rendering
#include "Module/GLPhotorealisticInstanceRender.h"


namespace dyno
{
	//Jeep
	IMPLEMENT_TCLASS(PresetArticulatedBody, TDataType)

	template<typename TDataType>
	PresetArticulatedBody<TDataType>::PresetArticulatedBody():
		ArticulatedBody<TDataType>()
	{

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&PresetArticulatedBody<TDataType>::varChanged, this));
		this->varFilePath()->attach(callback);

		this->inTextureMesh()->tagOptional(true);
		this->stateTextureMeshState()->setDataPtr(std::make_shared<TextureMesh>());


		auto transformer = this->graphicsPipeline()->findFirstModule<InstanceTransform<DataType3f>>();

		//this->stateCenter()->connect(transformer->inCenter());
		//this->stateInitialRotation()->connect(transformer->inInitialRotation());
		//this->stateRotationMatrix()->connect(transformer->inRotationMatrix());
		//this->stateBindingPair()->connect(transformer->inBindingPair());
		//this->stateBindingTag()->connect(transformer->inBindingTag());
		//this->stateInstanceTransform()->connect(transformer->inInstanceTransform());

		auto prRender = std::make_shared<GLPhotorealisticInstanceRender>();
		this->stateTextureMeshState()->connect(prRender->inTextureMesh());
		transformer->outInstanceTransform()->connect(prRender->inTransform());
		this->graphicsPipeline()->pushModule(prRender);
	}

	template<typename TDataType>
	PresetArticulatedBody<TDataType>::~PresetArticulatedBody()
	{

	}

	template<typename TDataType>
	void PresetArticulatedBody<TDataType>::varChanged()
	{
		std::shared_ptr<TextureMesh> texMesh = this->stateTextureMeshState()->getDataPtr();
		auto filepath = this->varFilePath()->getValue().string();
		loadGLTFTextureMesh(texMesh, filepath);



	}

	template<typename TDataType>
	void PresetArticulatedBody<TDataType>::resetStates()
	{
		ArticulatedBody<TDataType>::resetStates();
	}

	DEFINE_CLASS(PresetArticulatedBody);

	
	//Jeep
	IMPLEMENT_TCLASS(PresetJeep, TDataType)

		template<typename TDataType>
	PresetJeep<TDataType>::PresetJeep() :
		PresetArticulatedBody<TDataType>()
	{
		auto driver = std::make_shared<CarDriver<DataType3f>>();
		this->stateTopology()->connect(driver->inTopology());
		this->animationPipeline()->pushModule(driver);
		this->varFilePath()->setValue(FilePath(getAssetPath() + "Jeep/JeepGltf/jeep.gltf"));
	}

	template<typename TDataType>
	PresetJeep<TDataType>::~PresetJeep()
	{

	}

	template<typename TDataType>
	void PresetJeep<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		int vehicleNum = this->varVehiclesTransform()->getValue().size();
		for (size_t i = 0; i < vehicleNum; i++)
		{
			RigidBodyInfo rigidbody;
			rigidbody.bodyId = i;

			auto texMesh = this->stateTextureMeshState()->constDataPtr();

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

				rigidbody.position = texMesh->shapes()[it]->boundingTransform.translation();
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
				rigidbody.position = texMesh->shapes()[it]->boundingTransform.translation();
				actors[it] = this->addBox(boxs[it], rigidbody, 100);
			}

			//bindShapetoActor
			for (auto it : box_Id)
			{
				this->bind(actors[it], Pair<uint, uint>(it, i));
			}
			//bindShapetoActor
			for (auto it : Wheel_Id)
			{
				this->bind(actors[it], Pair<uint, uint>(it, i));
			}

			rigidbody.offset = Vec3f(0);

			Real wheel_velocity = 10;

			//wheel to Body
			for (auto it : Wheel_Id)
			{
				auto& joint = this->createHingeJoint(actors[it], actors[body]);
				joint.setAnchorPoint(actors[it]->center);
				joint.setMoter(wheel_velocity);
				joint.setAxis(Vec3f(1, 0, 0));
			}

			auto& jointBackWheel_Body = this->createFixedJoint(actors[backWheel], actors[body]);
			jointBackWheel_Body.setAnchorPoint(actors[backWheel]->center);		//set and offset
		}

		//**************************************************//
		PresetArticulatedBody<TDataType>::resetStates();
	}

	DEFINE_CLASS(PresetJeep);

	//Tank
	IMPLEMENT_TCLASS(PresetTank, TDataType)

		template<typename TDataType>
	PresetTank<TDataType>::PresetTank() :
		PresetArticulatedBody<TDataType>()
	{
		auto driver = std::make_shared<CarDriver<DataType3f>>();
		this->stateTopology()->connect(driver->inTopology());
		this->animationPipeline()->pushModule(driver);

		this->varFilePath()->setValue(FilePath(getAssetPath() + "gltf/Tank/Tank.gltf"));
	}

	template<typename TDataType>
	PresetTank<TDataType>::~PresetTank()
	{

	}

	template<typename TDataType>
	void PresetTank<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		int vehicleNum = this->varVehiclesTransform()->getValue().size();
		for (size_t i = 0; i < vehicleNum; i++)
		{
			RigidBodyInfo rigidbody;
			rigidbody.bodyId = i;

			auto texMesh = this->stateTextureMeshState()->constDataPtr();

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

				rigidbody.position = texMesh->shapes()[it]->boundingTransform.translation();
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
				rigidbody.position = texMesh->shapes()[it]->boundingTransform.translation();
				actors[it] = this->addBox(boxs[it], rigidbody, 100);
			}

			//bindShapetoActor
			for (auto it : box_Id)
			{
				this->bind(actors[it], Pair<uint, uint>(it, i));
			}
			//bindShapetoActor
			for (auto it : Wheel_Id)
			{
				this->bind(actors[it], Pair<uint, uint>(it, i));
			}

			rigidbody.offset = Vec3f(0);

			Real wheel_velocity = 10;

			//wheel to Body
			for (auto it : Wheel_Id)
			{
				auto& joint = this->createHingeJoint(actors[it], actors[body]);
				joint.setAnchorPoint(actors[it]->center);
				joint.setMoter(wheel_velocity);
				joint.setAxis(Vec3f(1, 0, 0));
			}

			auto& jointGun_Body = this->createFixedJoint(actors[gun], actors[body]);
			jointGun_Body.setAnchorPoint(actors[gun]->center);		//set and offset
		}

		//**************************************************//
		PresetArticulatedBody<TDataType>::resetStates();
	}

	DEFINE_CLASS(PresetTank);
}
