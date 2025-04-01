#include "ConfigurableBody.h"

#include "Module/CarDriver.h"

//Collision
#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionTriangleSet.h"

//RigidBody
#include "Module/ContactsUnion.h"
#include "Module/TJConstraintSolver.h"
#include "Module/InstanceTransform.h"
#include "Module/SharedFuncsForRigidBody.h"

//Rendering
#include "Module/GLPhotorealisticInstanceRender.h"

namespace dyno
{
	//ConfigurableVehicle
	IMPLEMENT_TCLASS(ConfigurableBody, TDataType)

		template<typename TDataType>
	ConfigurableBody<TDataType>::ConfigurableBody()
		: ParametricModel<TDataType>()
		, ArticulatedBody<TDataType>()
	{
		auto elementQuery = std::make_shared<NeighborElementQuery<TDataType>>();
		elementQuery->varSelfCollision()->setValue(false);
		this->stateTopology()->connect(elementQuery->inDiscreteElements());
		this->stateCollisionMask()->connect(elementQuery->inCollisionMask());
		this->stateAttribute()->connect(elementQuery->inAttribute());
		this->animationPipeline()->pushModule(elementQuery);

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

		this->inTriangleSet()->tagOptional(true);

		// 		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		// 		this->stateTopology()->connect(mapper->inDiscreteElements());
		// 		this->graphicsPipeline()->pushModule(mapper);
		// 
		// 		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		// 		sRender->setColor(Color(0.3f, 0.5f, 0.9f));
		// 		sRender->setAlpha(0.2f);
		// 		sRender->setRoughness(0.7f);
		// 		sRender->setMetallic(3.0f);
		// 		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		// 		this->graphicsPipeline()->pushModule(sRender);
		// 		sRender->setVisible(false);

		this->inTriangleSet()->tagOptional(true);
	}

	template<typename TDataType>
	ConfigurableBody<TDataType>::~ConfigurableBody()
	{

	}

	template<typename TDataType>
	void ConfigurableBody<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		if (!this->inTextureMesh()->isEmpty())
		{
			this->stateTextureMesh()->setDataPtr(this->inTextureMesh()->constDataPtr());
		}

		if (!this->varVehicleConfiguration()->getValue().isValid() && !bool(this->varVehiclesTransform()->getValue().size()) || this->stateTextureMesh()->isEmpty())
			return;

		auto texMesh = this->stateTextureMesh()->constDataPtr();
		const auto config = this->varVehicleConfiguration()->getValue();

		const auto rigidInfo = config.mVehicleRigidBodyInfo;
		const auto jointInfo = config.mVehicleJointInfo;

		// **************************** Create RigidBody  **************************** //
		auto instances = this->varVehiclesTransform()->getValue();
		uint vehicleNum = instances.size();
		int maxGroup = 0;
		for (size_t i = 0; i < rigidInfo.size(); i++)
		{
			if (rigidInfo[i].rigidGroup > maxGroup)
				maxGroup = rigidInfo[i].rigidGroup;
		}

		for (size_t j = 0; j < vehicleNum; j++)
		{
			

			std::vector<std::shared_ptr<PdActor>> Actors;

			Actors.resize(rigidInfo.size());


			for (size_t i = 0; i < rigidInfo.size(); i++)
			{
				RigidBodyInfo rigidbody;
				rigidbody.bodyId = j * (maxGroup + 1) + rigidInfo[i].rigidGroup;

				rigidbody.offset = rigidInfo[i].Offset;
				rigidbody.friction = this->varFrictionCoefficient()->getValue();

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
						
						rigidbody.position = Quat1f(instances[j].rotation()).rotate(T + rigidInfo[i].transform.translation()) + instances[j].translation();
						rigidbody.angle = Quat1f(instances[j].rotation());
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

						rigidbody.position = Quat1f(instances[j].rotation()).rotate(T + rigidInfo[i].transform.translation()) + instances[j].translation();
						rigidbody.angle = Quat1f(instances[j].rotation());
						Actors[i] = this->addCapsule(currentCapsule, rigidbody, density);
						break;

					case dyno::Sphere:
						currentSphere.center = Vec3f(0.0f);
						currentSphere.rot =  Quat1f(transform.rotation());
						currentSphere.radius = std::abs(up.y - down.y) / 2 * rigidInfo[i].radius;

						rigidbody.position = Quat1f(instances[j].rotation()).rotate(T + rigidInfo[i].transform.translation()) + instances[j].translation();
						rigidbody.angle = Quat1f(instances[j].rotation());
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

						rigidbody.position = Quat1f(instances[j].rotation()).rotate(rigidInfo[i].transform.translation()) + instances[j].translation();
						rigidbody.angle = Quat1f(instances[j].rotation());
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

						rigidbody.position = Quat1f(instances[j].rotation()).rotate(rigidInfo[i].transform.translation()) + instances[j].translation();
						rigidbody.angle = Quat1f(instances[j].rotation());
						Actors[i] = this->addCapsule(currentCapsule, rigidbody, density);
						break;

					case dyno::Sphere:
						currentSphere.center = Vec3f(0.0f);
						currentSphere.rot = Quat<Real>(rigidInfo[i].transform.rotation());
						currentSphere.radius = rigidInfo[i].radius;

						rigidbody.position = Quat1f(instances[j].rotation()).rotate(rigidInfo[i].transform.translation()) + instances[j].translation();
						rigidbody.angle = Quat1f(instances[j].rotation());
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
					this->bindShape(Actors[i], Pair<uint, uint>(shapeId, j));

				}
			}

			for (size_t i = 0; i < jointInfo.size(); i++)
			{
				////Actor
				auto type = jointInfo[i].mJointType;
				int first = jointInfo[i].mRigidBodyName_1.rigidBodyId;
				int second = jointInfo[i].mRigidBodyName_2.rigidBodyId;
				Real speed = jointInfo[i].mMoter;
				auto axis = Quat1f(instances[j].rotation()).rotate(jointInfo[i].mAxis);
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

		RigidBodySystem<TDataType>::postUpdateStates();

		this->updateInstanceTransform();
	}

	DEFINE_CLASS(ConfigurableBody);



}
