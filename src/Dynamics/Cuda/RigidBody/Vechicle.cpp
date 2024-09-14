#include "Vechicle.h"

#include "Module/SimpleVechicleDriver.h"
#include "Module/SharedFuncsForRigidBody.h"
#include "Module/ContactsUnion.h"
#include "Module/TJConstraintSolver.h"
#include "Module/TJSoftConstraintSolver.h"
#include "Module/PJSNJSConstraintSolver.h"
#include "Module/PJSoftConstraintSolver.h"
#include "Module/PJSConstraintSolver.h"
#include "Module/PCGConstraintSolver.h"
#include "Module/CarDriver.h"

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"
#include "Collision/CollistionDetectionTriangleSet.h"
#include <GLWireframeVisualModule.h>
#include <Mapping/ContactsToEdgeSet.h>

#include "Module/CarDriver.h"

#include "Module/CarDriver.h"

#include <Module/GLPhotorealisticInstanceRender.h>
#include <Mapping/DiscreteElementsToTriangleSet.h>
#include "GLSurfaceVisualModule.h"

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

		

// 		auto cdBV = std::make_shared<CollistionDetectionBoundingBox<TDataType>>();
// 		this->stateTopology()->connect(cdBV->inDiscreteElements());
// 		this->animationPipeline()->pushModule(cdBV);

		auto cdBV = std::make_shared<CollistionDetectionTriangleSet<TDataType>>();
		this->stateTopology()->connect(cdBV->inDiscreteElements());
		this->inTriangleSet()->connect(cdBV->inTriangleSet());
		// 		auto cdBV = std::make_shared<CollistionDetectionBoundingBox<TDataType>>();
		// 		this->stateTopology()->connect(cdBV->inDiscreteElements());
		this->animationPipeline()->pushModule(cdBV);


		auto merge = std::make_shared<ContactsUnion<TDataType>>();
		elementQuery->outContacts()->connect(merge->inContactsA());
		cdBV->outContacts()->connect(merge->inContactsB());
		this->animationPipeline()->pushModule(merge);

		auto iterSolver = std::make_shared<TJConstraintSolver<TDataType>>();
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

		/*auto driver = std::make_shared<SimpleVechicleDriver>();

		this->stateFrameNumber()->connect(driver->inFrameNumber());
		this->stateInstanceTransform()->connect(driver->inInstanceTransform());

		this->animationPipeline()->pushModule(driver);*/

		auto driver = std::make_shared<CarDriver<DataType3f>>();
		this->stateTopology()->connect(driver->inTopology());
		this->animationPipeline()->pushModule(driver);

		this->inTriangleSet()->tagOptional(true);


		auto prRender = std::make_shared<GLPhotorealisticInstanceRender>();
		this->inTextureMesh()->connect(prRender->inTextureMesh());
		this->stateInstanceTransform()->connect(prRender->inTransform());
		this->graphicsPipeline()->pushModule(prRender);

		this->setForceUpdate(true);
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

		uint N = 0;
		if(!this->inTextureMesh()->isEmpty())
			N = texMesh->shapes().size();

		CArrayList<Transform3f> tms;
		CArray<uint> instanceNum(N);
		instanceNum.reset();

		//Calculate instance number
		for (uint i = 0; i < mBindingPair.size(); i++)
		{
			instanceNum[mBindingPair[i].first]++;
		}

		if(instanceNum.size()>0)
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

		this->transform();
	}


	template<typename TDataType>
	void Vechicle<TDataType>::transform()
	{		
		////************************** initial mInitialRot *************************//

		mInitialRot.assign(this->stateRotationMatrix()->constData());

		CArray<Coord> hostCenter;
		hostCenter.assign(this->stateCenter()->constData());

		CArray<Quat<Real>> hostQuaternion;
		hostQuaternion.assign(this->stateQuaternion()->constData());

		CArray<Mat3f> hostRotation;
		hostRotation.assign(this->stateRotationMatrix()->constData());

		{	
			//get Elements
			auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->stateTopology()->getDataPtr());

			auto& boxes = topo->getBoxes();
			auto& spheres = topo->getSpheres();
			auto& tets = topo->getTets();
			auto& caps = topo->getCaps();

			std::vector<Transform3f> vehicleTransform = this->varVehiclesTransform()->getValue();

			int vehicleNum = vehicleTransform.size();



			for (size_t i = 0; i < vehicleNum; i++)
			{

				Quat<Real> q = Quat<Real>(vehicleTransform[i].rotation());

				Vec3f pos = vehicleTransform[i].translation();


				int boxesNum = boxes.size()/ vehicleNum;
				int spheresNum = spheres.size() / vehicleNum;
				int tetsNum = tets.size() / vehicleNum;
				int capsNum = caps.size() / vehicleNum;

				//***************************** Copy Translation *************************//
				for (uint j = 0; j < spheresNum; j++)
				{
					
					hostCenter[i * spheresNum + j] = q.rotate(hostCenter[i * spheresNum + j ]) + pos;
				}

				for (uint j = 0; j < boxesNum; j++)
				{
					int offset = spheres.size();
					hostCenter[i * boxesNum + j + offset] = q.rotate(hostCenter[i * boxesNum + j + offset]) + pos;
				}

				for (uint j = 0; j < tetsNum; j++)
				{
					int offset = boxes.size() + spheres.size();
					hostCenter[i * tetsNum + j + offset] = q.rotate(hostCenter[i * tetsNum + j + offset]) + pos;
				}

				for (uint j = 0; j < capsNum; j++)
				{
					int offset = boxes.size() + spheres.size() + tets.size();
					hostCenter[i * capsNum + j + offset] = q.rotate(hostCenter[i * capsNum + j + offset]) + pos;
				}

				//***************************** Copy Rotation *************************//

				for (uint j = 0; j < spheresNum; j++)
				{
					
					hostQuaternion[i * spheresNum + j] = q * hostQuaternion[i * spheresNum + j ];
				}
				for (uint j = 0; j < boxesNum; j++)
				{
					int offset = spheres.size();
					hostQuaternion[i * boxesNum + j + offset] = q * hostQuaternion[i * boxesNum + j + offset];
				}
				for (uint j = 0; j < tetsNum; j++)
				{
					int offset = boxes.size() + spheres.size();
					hostQuaternion[i * tetsNum + j + offset] = q * hostQuaternion[i * tetsNum + j + offset];
				}
				for (uint j = 0; j < capsNum; j++)
				{
					int offset = boxes.size() + spheres.size() + tets.size();
					hostQuaternion[i * capsNum + j + offset] = q * hostQuaternion[i * capsNum + j + offset];
				}


				for (uint j = 0; j < spheresNum; j++)
				{			
					hostRotation[i * spheresNum + j] = q.toMatrix3x3() * hostRotation[i * spheresNum + j ];
				}
				for (uint j = 0; j < boxesNum; j++)
				{
					int offset = spheres.size();
					hostRotation[i * boxesNum + j + offset] = q.toMatrix3x3() * hostRotation[i * boxesNum + j + offset];
				}
				for (uint j = 0; j < tetsNum; j++)
				{
					int offset = boxes.size() + spheres.size();
					hostRotation[i * tetsNum + j + offset] = q.toMatrix3x3() * hostRotation[i * tetsNum + j + offset];
				}
				for (uint j = 0; j < capsNum; j++)
				{
					int offset = boxes.size() + spheres.size() + tets.size();
					hostRotation[i * capsNum + j + offset] = q.toMatrix3x3() * hostRotation[i * capsNum + j + offset];
				}


			}

		}


		{
			//get varTransform;
			auto quat = this->computeQuaternion();
			Coord location = this->varLocation()->getValue();

			//***************************** Translation *************************//


			for (uint i = 0; i < hostCenter.size(); i++)
			{
				hostCenter[i] = quat.rotate(hostCenter[i]) + location;
			}

			//***************************** Rotation *************************//

			for (uint i = 0; i < hostQuaternion.size(); i++)
			{
				hostQuaternion[i] = quat * hostQuaternion[i];
			}

			for (uint i = 0; i < hostRotation.size(); i++)
			{
				hostRotation[i] = quat.toMatrix3x3() * hostRotation[i];
			}



		}
		
		this->stateCenter()->assign(hostCenter);
		this->stateQuaternion()->assign(hostQuaternion);
		this->stateRotationMatrix()->assign(hostRotation);

		hostCenter.clear();
		hostQuaternion.clear();
		hostRotation.clear();

		this->updateTopology();

		this->updateInstanceTransform();
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

	template<typename TDataType>
	void Vechicle<TDataType>::clearVechicle()
	{
		mBindingPair.clear();
		mBindingPairDevice.clear();
		mBindingTagDevice.clear();
		mInitialRot.clear();
		mActors.clear();
	}

	DEFINE_CLASS(Vechicle);

	//Jeep
	IMPLEMENT_TCLASS(Jeep, TDataType)

	template<typename TDataType>
	Jeep<TDataType>::Jeep()
		: ParametricModel<TDataType>()
		, Vechicle<TDataType>()
	{
		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		this->stateTopology()->connect(mapper->inDiscreteElements());
		this->graphicsPipeline()->pushModule(mapper);
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
			std::map<int, std::shared_ptr<PdActor>> Actors;
			//Capsule
			for (auto it : Wheel_Id)
			{
				auto up = texMesh->shapes()[it]->boundingBox.v1;
				auto down = texMesh->shapes()[it]->boundingBox.v0;

				wheels[it].center = texMesh->shapes()[it]->boundingTransform.translation();
				wheels[it].rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
				wheels[it].halfLength = 0.1;
				wheels[it].radius = std::abs(up.y - down.y) / 2;
			}
			//Actor
			for (auto it : Wheel_Id)
			{
				Actors[it] = this->addCapsule(wheels[it], rigidbody, 100);
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

				auto center = texMesh->shapes()[it]->boundingTransform.translation();
				boxs[it].center = Vec3f(center.x, center.y, center.z);

				boxs[it].halfLength = (up - down) / 2;


			}
			//Actor
			for (auto it : box_Id)
			{
				Vec3f offset = Vec3f(0.0f, 0.0f, 0.0f);

				rigidbody.offset = offset;
				Actors[it] = this->addBox(boxs[it], rigidbody, 100);
			}
			//bindShapetoActor
			for (auto it : box_Id)
			{
				this->bind(Actors[it], Pair<uint, uint>(it, i));
			}
			//bindShapetoActor
			for (auto it : Wheel_Id)
			{
				this->bind(Actors[it], Pair<uint, uint>(it, i));
			}

			rigidbody.offset = Vec3f(0);

			Real wheel_velocity = 10;

			//wheel to Body
			for (auto it : Wheel_Id)
			{
				auto& joint = this->createHingeJoint(Actors[it], Actors[body]);
				joint.setAnchorPoint(Actors[it]->center);
				joint.setMoter(wheel_velocity);
				joint.setAxis(Vec3f(1, 0, 0));
			}

			auto& jointBackWheel_Body = this->createFixedJoint(Actors[backWheel], Actors[body]);
			jointBackWheel_Body.setAnchorPoint(Actors[backWheel]->center);		//set and offset

		}

		//**************************************************//
		Vechicle<TDataType>::resetStates();


		this->updateTopology();

		this->updateInstanceTransform();

		
	}

	DEFINE_CLASS(Jeep);



	//ConfigurableVehicle
	IMPLEMENT_TCLASS(ConfigurableVehicle, TDataType)

	template<typename TDataType>
	ConfigurableVehicle<TDataType>::ConfigurableVehicle()
		: ParametricModel<TDataType>()
		, Vechicle<TDataType>()
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
		for (size_t j = 0; j < vehicleNum; j++)
		{
			RigidBodyInfo rigidbody;
			rigidbody.bodyId = j;
			std::vector<std::shared_ptr<PdActor>> Actors;

			Actors.resize(rigidInfo.size());

			for (size_t i = 0; i < rigidInfo.size(); i++)
			{

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
						currentBox.center = T + rigidInfo[i].transform.translation();;
						currentBox.halfLength = (up - down) / 2 * rigidInfo[i].mHalfLength;
						currentBox.rot = Quat1f(transform.rotation());

						Actors[i] = this->addBox(currentBox, rigidbody, density);
						break;

					case dyno::Tet:
						printf("Need Tet Configuration\n");
						break;

					case dyno::Capsule:
						currentCapsule.center = T + rigidInfo[i].transform.translation();
						currentCapsule.rot = Quat1f(transform.rotation());
						currentCapsule.halfLength = (up.y - down.y) / 2 * rigidInfo[i].capsuleLength;
						currentCapsule.radius = std::abs(up.y - down.y) / 2 * rigidInfo[i].radius;

						Actors[i] = this->addCapsule(currentCapsule, rigidbody, density);
						break;

					case dyno::Sphere:
						currentSphere.center = T + rigidInfo[i].transform.translation();
						currentSphere.rot = Quat1f(transform.rotation());
						currentSphere.radius = std::abs(up.y - down.y) / 2 * rigidInfo[i].radius;

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
						currentBox.center = rigidInfo[i].transform.translation();
						currentBox.halfLength = rigidInfo[i].mHalfLength;
						currentBox.rot = Quat<Real>(rigidInfo[i].transform.rotation());

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
						currentCapsule.center = rigidInfo[i].transform.translation();
						currentCapsule.rot = Quat<Real>(rigidInfo[i].transform.rotation());
						currentCapsule.halfLength = rigidInfo[i].capsuleLength;
						currentCapsule.radius = rigidInfo[i].radius;

						Actors[i] = this->addCapsule(currentCapsule, rigidbody, density);
						break;

					case dyno::Sphere:
						currentSphere.center = rigidInfo[i].transform.translation();
						currentSphere.rot = Quat<Real>(rigidInfo[i].transform.rotation());
						currentSphere.radius = rigidInfo[i].radius;

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
		Vechicle<TDataType>::resetStates();


		this->updateTopology();

		this->updateInstanceTransform();

	}

	DEFINE_CLASS(ConfigurableVehicle);
}
