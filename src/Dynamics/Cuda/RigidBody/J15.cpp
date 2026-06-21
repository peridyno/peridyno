#include "J15.h"

#include "Mapping/DiscreteElementsToTriangleSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

//Modeling
#include "GltfFunc.h"

//Rigidbody
#include "RigidBody/Module/InstanceTransform.h"

//Rendering
#include "Module/GLPhotorealisticInstanceRender.h"

#include <cmath>
#include <vector>

namespace dyno
{
	IMPLEMENT_TCLASS(J15Operator, TDataType);

	template<typename TDataType>
	J15Operator<TDataType>::J15Operator()
		: KeyboardInputModule()
	{
		this->setUpdateAlways(true);
		this->varCacheEvent()->setValue(false);
		this->inTimeStep()->setValue(Real(0.016));
	}

	template<typename TDataType>
	typename J15Operator<TDataType>::Real J15Operator<TDataType>::clamp(Real value, Real lo, Real hi)
	{
		return std::max(lo, std::min(value, hi));
	}

	template<typename TDataType>
	void J15Operator<TDataType>::setKeyState(PKeyboardType key, bool pressed)
	{
		switch (key)
		{
		case PKeyboardType::PKEY_A:
			mKeyA = pressed;
			break;
		case PKeyboardType::PKEY_S:
			mKeyS = pressed;
			break;
		case PKeyboardType::PKEY_D:
			mKeyD = pressed;
			break;
		case PKeyboardType::PKEY_W:
			mKeyW = pressed;
			break;
		case PKeyboardType::PKEY_Q:
			mKeyQ = pressed;
			break;
		case PKeyboardType::PKEY_E:
			mKeyE = pressed;
			break;
		case PKeyboardType::PKEY_Z:
			mKeyZ = pressed;
			break;
		case PKeyboardType::PKEY_X:
			mKeyX = pressed;
			break;
		case PKeyboardType::PKEY_J:
			mKeyJ = pressed;
			break;
		case PKeyboardType::PKEY_K:
			mKeyK = pressed;
			break;
		case PKeyboardType::PKEY_V:
			mKeyV = pressed;
			break;
		default:
			break;
		}
	}

	template<typename TDataType>
	void J15Operator<TDataType>::reset()
	{
		mState.LinearVelocity = Coord(0.0);
		mState.LinearVelocityIncrement = Coord(0.0);
		mState.Attitude.Roll = Real(0);
		mState.Attitude.Pitch = Real(0);
		mState.Attitude.Yaw = Real(0);
		mBodyVelocityIncrement = Coord(0);
		mRollIncrement = Real(0);
		mPitchIncrement = Real(0);
		mElevatorPitchIncrement = Real(0);
		mYawIncrement = Real(0);
		mHasAttitudeIncrement = false;
		mHasVelocityIncrement = false;

		mInput.Elevator = Real(0);
		mInput.Aileron = Real(0);
		mInput.Rudder = Real(0);
		mInput.Stabilator = Real(0);
		mInput.Throttle = Real(0);

		mKeyA = false;
		mKeyS = false;
		mKeyD = false;
		mKeyW = false;
		mKeyQ = false;
		mKeyE = false;
		mKeyZ = false;
		mKeyX = false;
		mKeyJ = false;
		mKeyK = false;
		mKeyV = false;
	}

	template<typename TDataType>
	void J15Operator<TDataType>::onEvent(PKeyboardEvent event)
	{
		if (event.action == PActionType::AT_PRESS || event.action == PActionType::AT_REPEAT)
		{
			this->setKeyState(event.key, true);
			if (event.key == PKeyboardType::PKEY_V)
			{
				this->reset();
				mResetRequested = true;
			}
		}
		else if (event.action == PActionType::AT_RELEASE)
		{
			this->setKeyState(event.key, false);
		}
	}

	template<typename TDataType>
	void J15Operator<TDataType>::postprocess()
	{
		if (this->inCenter()->isEmpty() || this->inQuaternion()->isEmpty() || this->inRotationMatrix()->isEmpty())
			return;

		//mBodyId = this->inCenter()->size() > 5 ? 5 : 0;
		//mBodyId = 5;

		std::cout << "mBodyId" << mBodyId << std::endl;

		if (!mInitialized)
		{
			CArray<TQuat> quaternions;
			quaternions.assign(this->inQuaternion()->constData());

			mInitialQuaternion = quaternions[mBodyId];

			if (!this->inVelocity()->isEmpty() && this->inVelocity()->size() > mBodyId)
			{
				CArray<Coord> velocities;
				velocities.assign(this->inVelocity()->constData());
				mState.LinearVelocity = velocities[mBodyId];
				velocities.clear();
			}

			quaternions.clear();
			mInitialized = true;
		}

		if (!mResetRequested)
		{
			this->updateInput();
			this->stepFlightModel(std::max(this->inTimeStep()->getValue(), Real(0)));
		}
		this->applyFlightState();
	}

	template<typename TDataType>
	void J15Operator<TDataType>::updateInput()
	{
		if (mExternalInput)
			return;

		const Real step = Real(0.01);

		if (mKeyA)
			mInput.Rudder = clamp(mInput.Rudder + step, Real(-1), Real(1));
		if (mKeyD)
			mInput.Rudder = clamp(mInput.Rudder - step, Real(-1), Real(1));

		if (mKeyW)
			mInput.Elevator = clamp(mInput.Elevator + step, Real(-1), Real(1));
		if (mKeyS)
			mInput.Elevator = clamp(mInput.Elevator - step, Real(-1), Real(1));

		if (mKeyQ)
			mInput.Aileron = clamp(mInput.Aileron + step, Real(-1), Real(1));
		if (mKeyE)
			mInput.Aileron = clamp(mInput.Aileron - step, Real(-1), Real(1));

		if (mKeyZ)
			mInput.Stabilator = clamp(mInput.Stabilator + step, Real(-1), Real(1));
		if (mKeyX)
			mInput.Stabilator = clamp(mInput.Stabilator - step, Real(-1), Real(1));

		if (mKeyJ)
			mInput.Throttle = clamp(mInput.Throttle + step, Real(0), Real(1));
		if (mKeyK)
			mInput.Throttle = clamp(mInput.Throttle - step, Real(0), Real(1));
	}

	template<typename TDataType>
	void J15Operator<TDataType>::stepFlightModel(Real dt)
	{
		const Real pitchCommand = mInput.Elevator + mConstants.StabilatorWeight * mInput.Stabilator;
		const Real oldRoll = mState.Attitude.Roll;
		const Real oldPitch = mState.Attitude.Pitch;

		Real nextRoll = oldRoll + mConstants.RollGain * mInput.Aileron * dt;
		const Real rawElevatorPitchIncrement = mConstants.PitchGain * mInput.Elevator * dt;
		const Real rawPitchIncrement = mConstants.PitchGain * pitchCommand * dt;
		Real nextPitch = oldPitch + rawPitchIncrement;

		if (mConstants.MaxAbsRoll > Real(0))
		{
			nextRoll = clamp(nextRoll, -mConstants.MaxAbsRoll, mConstants.MaxAbsRoll);
		}

		if (mConstants.MaxAbsPitch > Real(0))
		{
			nextPitch = clamp(nextPitch, -mConstants.MaxAbsPitch, mConstants.MaxAbsPitch);
		}

		mRollIncrement = nextRoll - oldRoll;
		mPitchIncrement = nextPitch - oldPitch;
		mElevatorPitchIncrement = Real(0);
		if (std::abs(rawPitchIncrement) > Real(1e-7))
		{
			mElevatorPitchIncrement = rawElevatorPitchIncrement * mPitchIncrement / rawPitchIncrement;
		}
		mYawIncrement = mConstants.YawGain * mInput.Rudder * dt;
		mHasAttitudeIncrement = std::abs(mRollIncrement) > Real(1e-7)
			|| std::abs(mPitchIncrement) > Real(1e-7)
			|| std::abs(mYawIncrement) > Real(1e-7);

		mState.Attitude.Roll = nextRoll;
		mState.Attitude.Pitch = nextPitch;
		mState.Attitude.Yaw += mYawIncrement;

		mState.LinearVelocityIncrement = Coord(0);
		mBodyVelocityIncrement = Coord(0);

		const Real forwardAcceleration = mConstants.ThrottleGain * mInput.Throttle - mConstants.DragAcceleration;
		mBodyVelocityIncrement = Coord(
			forwardAcceleration,
			mConstants.LiftGain * pitchCommand,
			Real(0)) * dt;
		mHasVelocityIncrement = mBodyVelocityIncrement.dot(mBodyVelocityIncrement) > Real(1e-12);
	}

	template<typename TDataType>
	void J15Operator<TDataType>::applyFlightState()
	{
		const uint bodyNum = this->inCenter()->size();
		if (bodyNum == 0 || mBodyId >= bodyNum)
			return;

		const bool attitudeDirty = mResetRequested || mHasAttitudeIncrement;
		const bool velocityDirty = mResetRequested || mHasVelocityIncrement || mHasAttitudeIncrement;
		if (!attitudeDirty && !velocityDirty)
		{
			return;
		}

		CArray<TQuat> quaternions;
		quaternions.assign(this->inQuaternion()->constData());

		const bool updatePitchPivot = !mResetRequested && std::abs(mElevatorPitchIncrement) > Real(1e-7);
		CArray<Coord> centers;
		if (updatePitchPivot)
		{
			centers.assign(this->inCenter()->constData());
		}

		CArray<Matrix> rotations;
		if (attitudeDirty)
		{
			rotations.assign(this->inRotationMatrix()->constData());
		}

		CArray<Coord> velocities;
		const bool updateVelocity = velocityDirty && !this->inVelocity()->isEmpty() && this->inVelocity()->size() == bodyNum;
		if (updateVelocity)
		{
			velocities.assign(this->inVelocity()->constData());
		}

		CArray<Coord> angularVelocities;
		const bool updateAngularVelocity = mResetRequested && !this->inAngularVelocity()->isEmpty() && this->inAngularVelocity()->size() == bodyNum;
		if (updateAngularVelocity)
		{
			angularVelocities.assign(this->inAngularVelocity()->constData());
		}

		CArray<Matrix> inertias;
		CArray<Matrix> initialInertias;
		const bool updateInertia = attitudeDirty
			&& !this->inInertia()->isEmpty()
			&& !this->inInitialInertia()->isEmpty()
			&& this->inInertia()->size() == bodyNum
			&& this->inInitialInertia()->size() == bodyNum;
		if (updateInertia)
		{
			inertias.assign(this->inInertia()->constData());
			initialInertias.assign(this->inInitialInertia()->constData());
		}

		const TQuat oldBodyQuat = quaternions[mBodyId];
		TQuat increment = TQuat(Real(0), Real(0), Real(0), Real(1));
		TQuat elevatorPitchRotation = TQuat(Real(0), Real(0), Real(0), Real(1));
		Coord pitchPivot = Coord(0);
		if (mResetRequested)
		{
			increment = mInitialQuaternion * oldBodyQuat.inverse();
			increment.normalize();
		}
		else if (mHasAttitudeIncrement)
		{
			const Coord forwardAxis = oldBodyQuat.rotate(Coord(1, 0, 0));
			const Coord rightAxis = oldBodyQuat.rotate(Coord(0, 0, -1));
			const Coord upAxis = oldBodyQuat.rotate(Coord(0, 1, 0));

			if (updatePitchPivot)
			{
				const Coord tailPivotLocalOffset = Coord(Real(-9.0), Real(-0.5), Real(0));
				pitchPivot = centers[mBodyId] + oldBodyQuat.rotate(tailPivotLocalOffset);
				elevatorPitchRotation = TQuat(mElevatorPitchIncrement, rightAxis);
				elevatorPitchRotation.normalize();
			}

			if (std::abs(mYawIncrement) > Real(0))
			{
				increment = TQuat(mYawIncrement, upAxis) * increment;
			}
			if (std::abs(mPitchIncrement) > Real(0))
			{
				increment = TQuat(mPitchIncrement, rightAxis) * increment;
			}
			if (std::abs(mRollIncrement) > Real(0))
			{
				increment = TQuat(mRollIncrement, forwardAxis) * increment;
			}
			increment.normalize();
		}

		TQuat velocityQuat = attitudeDirty ? increment * oldBodyQuat : oldBodyQuat;
		velocityQuat.normalize();
		mState.LinearVelocityIncrement = velocityQuat.rotate(mBodyVelocityIncrement);
		const Coord forwardAxis = velocityQuat.rotate(Coord(1, 0, 0));
		const Real velocityCorrectionRate = clamp(
			mConstants.VelocityDirectionCorrectionGain * std::max(this->inTimeStep()->getValue(), Real(0)),
			Real(0),
			Real(1));

		for (uint i = 0; i < bodyNum; i++)
		{
			if (updatePitchPivot)
			{
				centers[i] = pitchPivot + elevatorPitchRotation.rotate(centers[i] - pitchPivot);
			}

			if (attitudeDirty)
			{
				quaternions[i] = increment * quaternions[i];
				quaternions[i].normalize();
				rotations[i] = quaternions[i].toMatrix3x3();
			}

			if (updateVelocity)
			{
				if (mResetRequested)
				{
					velocities[i] = Coord(0);
				}
				else
				{
					velocities[i] += mState.LinearVelocityIncrement;
					const Real speed = velocities[i].norm();
					if (speed > Real(1e-7))
					{
						const Coord targetVelocity = forwardAxis * speed;
						velocities[i] += (targetVelocity - velocities[i]) * velocityCorrectionRate;
					}
					else
					{
						velocities[i] = Coord(0);
					}
				}
			}

			if (mResetRequested && updateAngularVelocity)
			{
				angularVelocities[i] = Coord(0);
			}

			if (updateInertia)
			{
				inertias[i] = rotations[i] * initialInertias[i] * rotations[i].inverse();
			}
		}

		if (attitudeDirty)
		{
			if (updatePitchPivot)
			{
				this->inCenter()->assign(centers);
			}
			this->inQuaternion()->assign(quaternions);
			this->inRotationMatrix()->assign(rotations);
		}

		if (updateVelocity)
		{
			mState.LinearVelocity = velocities[mBodyId];
			this->inVelocity()->assign(velocities);
		}

		if (updateAngularVelocity)
		{
			this->inAngularVelocity()->assign(angularVelocities);
		}

		if (updateInertia)
		{
			this->inInertia()->assign(inertias);
		}

		auto topo = this->inTopology()->getDataPtr();
		if (topo != nullptr)
		{
			if (attitudeDirty)
			{
				topo->setPosition(this->inCenter()->constData());
				topo->setRotation(this->inRotationMatrix()->constData());
				topo->update();
			}
		}

		centers.clear();
		quaternions.clear();
		rotations.clear();
		velocities.clear();
		angularVelocities.clear();
		inertias.clear();
		initialInertias.clear();

		mResetRequested = false;
		mHasAttitudeIncrement = false;
		mHasVelocityIncrement = false;
	}

	DEFINE_CLASS(J15Operator);

	//J15
	IMPLEMENT_TCLASS(J15, TDataType)

	template<typename TDataType>
	J15<TDataType>::J15() :
		ArticulatedBody<TDataType>()
	{
		aircraftOperator = std::make_shared<J15Operator<TDataType>>();
		this->stateTimeStep()->connect(aircraftOperator->inTimeStep());
		this->stateCenter()->connect(aircraftOperator->inCenter());
		this->stateVelocity()->connect(aircraftOperator->inVelocity());
		this->stateAngularVelocity()->connect(aircraftOperator->inAngularVelocity());
		this->stateQuaternion()->connect(aircraftOperator->inQuaternion());
		this->stateRotationMatrix()->connect(aircraftOperator->inRotationMatrix());
		this->stateInertia()->connect(aircraftOperator->inInertia());
		this->stateInitialInertia()->connect(aircraftOperator->inInitialInertia());
		this->stateTopology()->connect(aircraftOperator->inTopology());
		this->animationPipeline()->pushModule(aircraftOperator);

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
		if (aircraftOperator != nullptr)
		{
			aircraftOperator->reset();
		}

		this->clearRigidBodySystem();
		this->clearVechicle();

		std::string filename = getAssetPath() + "J15/J15.obj";
		if (this->varFilePath()->getValue() != filename)
		{
			this->varFilePath()->setValue(FilePath(filename));
		}

		auto instances = this->varVehiclesTransform();
		int i = 0;
		for (auto it = instances->begin(); it != instances->end(); it++,i++)
		{
			auto instance = instances->getElement(it);
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

				rigidbody.position = Quat1f(instance.rotation()).rotate(texMesh->shapes()[id]->boundingTransform.translation()) + instance.translation();
				rigidbody.angle = Quat1f(instance.rotation());

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
				rigidbody.position = Quat1f(instance.rotation()).rotate(texMesh->shapes()[id]->boundingTransform.translation()) + instance.translation();
				rigidbody.angle = Quat1f(instance.rotation());

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

				rigidbody.position = Quat1f(instance.rotation()).rotate(texMesh->shapes()[bodyId]->boundingTransform.translation()) + instance.translation();
				rigidbody.angle = Quat1f(instance.rotation());

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
				rigidbody.position = Quat1f(instance.rotation()).rotate(texMesh->shapes()[fCover]->boundingTransform.translation()) + instance.translation();
				rigidbody.angle = Quat1f(instance.rotation());

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
				//wheelJoint.setMoter(-10);

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
