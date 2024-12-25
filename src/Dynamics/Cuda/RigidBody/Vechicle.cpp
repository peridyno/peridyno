#include "Vechicle.h"

#include "Module/CarDriver.h"

#include "Mapping/DiscreteElementsToTriangleSet.h"
#include "GLSurfaceVisualModule.h"

namespace dyno
{
	//Jeep
	IMPLEMENT_TCLASS(Jeep, TDataType)

	template<typename TDataType>
	Jeep<TDataType>::Jeep()
		: ParametricModel<TDataType>()
		, ArticulatedBody<TDataType>()
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

		int vehicleNum = this->varVehiclesTransform()->getValue().size();
		for (size_t i = 0; i < vehicleNum; i++)
		{	
			RigidBodyInfo rigidbody;
			rigidbody.bodyId = i;

			auto texMesh = this->inTextureMesh()->constDataPtr();

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
		ArticulatedBody<TDataType>::resetStates();
	}

	DEFINE_CLASS(Jeep);



	//ConfigurableVehicle
	IMPLEMENT_TCLASS(ConfigurableVehicle, TDataType)

	template<typename TDataType>
	ConfigurableVehicle<TDataType>::ConfigurableVehicle()
		: ParametricModel<TDataType>()
		, ArticulatedBody<TDataType>()
	{
		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		this->stateTopology()->connect(mapper->inDiscreteElements());
		this->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->setColor(Color(0.3f, 0.5f, 0.9f));
		sRender->setAlpha(0.2f);
		sRender->setRoughness(0.7f);
		sRender->setMetallic(3.0f);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);
		sRender->setVisible(false);

		this->inTriangleSet()->tagOptional(true);
		this->inTextureMesh()->tagOptional(true);
	}

	template<typename TDataType>
	ConfigurableVehicle<TDataType>::~ConfigurableVehicle()
	{

	}

	template<typename TDataType>
	void ConfigurableVehicle<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		

		if (!this->varVehicleConfiguration()->getValue().isValid()&& !bool(this->varVehiclesTransform()->getValue().size())|| this->inTextureMesh()->isEmpty())
			return;

		
		auto texMesh = this->inTextureMesh()->constDataPtr();
		const auto config = this->varVehicleConfiguration()->getValue();		

		const auto rigidInfo = config.mVehicleRigidBodyInfo;
		const auto jointInfo = config.mVehicleJointInfo;

		// **************************** Create RigidBody  **************************** //
		int vehicleNum = this->varVehiclesTransform()->getValue().size();
		int maxGroup = 0;
		for (size_t i = 0; i < rigidInfo.size(); i++)
		{
			if (rigidInfo[i].rigidGroup > maxGroup)
				maxGroup = rigidInfo[i].rigidGroup;
		}

		for (size_t j = 0; j < vehicleNum; j++)
		{
			RigidBodyInfo rigidbody;

			std::vector<std::shared_ptr<PdActor>> Actors;

			Actors.resize(rigidInfo.size());


			for (size_t i = 0; i < rigidInfo.size(); i++)
			{
				rigidbody.bodyId = j * (maxGroup+1) + rigidInfo[i].rigidGroup;

				rigidbody.offset = rigidInfo[i].Offset;

				auto type = rigidInfo[i].shapeType;
				auto shapeId = rigidInfo[i].meshShapeId;
				auto transform = rigidInfo[i].transform;
				Real density = rigidInfo[i].mDensity;

				if (shapeId > texMesh->shapes().size() - 1)
					continue;

				Vec3f up;
				Vec3f down;
				Vec3f T;

				if (shapeId != -1)
				{
					up = texMesh->shapes()[shapeId]->boundingBox.v1;
					down = texMesh->shapes()[shapeId]->boundingBox.v0;
					T = texMesh->shapes()[shapeId]->boundingTransform.translation();
				}
				else
				{


				}

				BoxInfo currentBox;
				CapsuleInfo currentCapsule;
				SphereInfo currentSphere;
				TetInfo currentTet;


				if (shapeId != -1)
				{
					switch (type)
					{
					case dyno::Box:
						currentBox.center = Vec3f(0.0f);
						currentBox.halfLength = (up - down) / 2 * rigidInfo[i].mHalfLength;
						currentBox.rot = Quat1f(transform.rotation());

						rigidbody.position = T + rigidInfo[i].transform.translation();
						Actors[i] = this->addBox(currentBox, rigidbody, density);
						break;

					case dyno::Tet:
						printf("Need Tet Configuration\n");
						break;

					case dyno::Capsule:
						currentCapsule.center = Vec3f(0.0f);
						currentCapsule.rot = Quat1f(transform.rotation());
						currentCapsule.halfLength = (up.y - down.y) / 2 * rigidInfo[i].capsuleLength;
						currentCapsule.radius = std::abs(up.y - down.y) / 2 * rigidInfo[i].radius;

						rigidbody.position = T + rigidInfo[i].transform.translation();
						Actors[i] = this->addCapsule(currentCapsule, rigidbody, density);
						break;

					case dyno::Sphere:
						currentSphere.center = Vec3f(0.0f);
						currentSphere.rot = Quat1f(transform.rotation());
						currentSphere.radius = std::abs(up.y - down.y) / 2 * rigidInfo[i].radius;

						rigidbody.position = T + rigidInfo[i].transform.translation();
						Actors[i] = this->addSphere(currentSphere, rigidbody, density);
						break;

					case dyno::Tri:
						printf("Need Tri Configuration\n");
						Actors[i] = NULL;
						break;

					case dyno::OtherShape:
						printf("Need OtherShape Configuration\n");
						Actors[i] = NULL;
						break;

					default:
						break;
					}
				}
				else if (shapeId == -1)
				{
					switch (type)
					{
					case dyno::Box:
						currentBox.center = Vec3f(0.0f);
						currentBox.halfLength = rigidInfo[i].mHalfLength;
						currentBox.rot = Quat<Real>(rigidInfo[i].transform.rotation());

						rigidbody.position = rigidInfo[i].transform.translation();
						Actors[i] = this->addBox(currentBox, rigidbody, density);
						break;

					case dyno::Tet:
						printf("Need Tet Configuration\n");
						currentTet.v[0] = rigidInfo[i].tet[0];
						currentTet.v[1] = rigidInfo[i].tet[1];
						currentTet.v[2] = rigidInfo[i].tet[2];
						currentTet.v[3] = rigidInfo[i].tet[3];

						break;

					case dyno::Capsule:
						currentCapsule.center = Vec3f(0.0f);
						currentCapsule.rot = Quat<Real>(rigidInfo[i].transform.rotation());
						currentCapsule.halfLength = rigidInfo[i].capsuleLength;
						currentCapsule.radius = rigidInfo[i].radius;

						rigidbody.position = rigidInfo[i].transform.translation();
						Actors[i] = this->addCapsule(currentCapsule, rigidbody, density);
						break;

					case dyno::Sphere:
						currentSphere.center = Vec3f(0.0f);
						currentSphere.rot = Quat<Real>(rigidInfo[i].transform.rotation());
						currentSphere.radius = rigidInfo[i].radius;

						rigidbody.position = rigidInfo[i].transform.translation();
						Actors[i] = this->addSphere(currentSphere, rigidbody, density);
						break;

					case dyno::Tri:
						printf("Need Tri Configuration\n");
						break;

					case dyno::OtherShape:
						printf("Need OtherShape Configuration\n");
						break;

					default:
						break;
					}
				}

				if (shapeId != -1 && Actors[i] != NULL)
				{
					////bindShapetoActor
					this->bind(Actors[i], Pair<uint, uint>(shapeId, j));

				}
			}

			for (size_t i = 0; i < jointInfo.size(); i++)
			{
				////Actor
				auto type = jointInfo[i].mJointType;
				int first = jointInfo[i].mRigidBodyName_1.rigidBodyId;
				int second = jointInfo[i].mRigidBodyName_2.rigidBodyId;
				Real speed = jointInfo[i].mMoter;
				auto axis = jointInfo[i].mAxis;
				auto anchorOffset = jointInfo[i].mAnchorPoint;

				if (first == -1 || second == -1)
					continue;
				if (Actors[first] == NULL || Actors[second] == NULL)
					continue;


				if (type == Hinge)
				{
					auto& joint = this->createHingeJoint(Actors[first], Actors[second]);
					joint.setAnchorPoint(Actors[first]->center + anchorOffset);
					joint.setAxis(axis);
					if (jointInfo[i].mUseMoter)
						joint.setMoter(speed);
					if (jointInfo[i].mUseRange)
						joint.setRange(jointInfo[i].mMin, jointInfo[i].mMax);

				}
				if (type == Slider)
				{
					auto& sliderJoint = this->createSliderJoint(Actors[first], Actors[second]);
					sliderJoint.setAnchorPoint((Actors[first]->center + Actors[first]->center) / 2 + anchorOffset);
					sliderJoint.setAxis(axis);
					if (jointInfo[i].mUseMoter)
						sliderJoint.setMoter(speed);
					if (jointInfo[i].mUseRange)
						sliderJoint.setRange(jointInfo[i].mMin, jointInfo[i].mMax);
				}
				if (type == Fixed)
				{
					auto& fixedJoint1 = this->createFixedJoint(Actors[first], Actors[second]);
					fixedJoint1.setAnchorPoint((Actors[first]->center + Actors[first]->center) / 2 + anchorOffset);
				}
				if (type == Point)
				{
					auto& pointJoint = this->createPointJoint(Actors[first]);
					pointJoint.setAnchorPoint(Actors[first]->center + anchorOffset);
				}
				if (type == BallAndSocket)
				{
					auto& ballAndSocketJoint = this->createBallAndSocketJoint(Actors[first], Actors[second]);
					ballAndSocketJoint.setAnchorPoint((Actors[first]->center + Actors[first]->center) / 2 + anchorOffset);
				}

				//Joint(Actor)
		
				


			}
		}

		
		/***************** Reset *************/
		ArticulatedBody<TDataType>::resetStates();

		this->updateInstanceTransform();
	}

	DEFINE_CLASS(ConfigurableVehicle);
}
