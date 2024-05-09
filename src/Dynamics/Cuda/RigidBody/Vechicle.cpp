#include "Vechicle.h"

#include "Module/SimpleVechicleDriver.h"
#include "Module/SharedFuncsForRigidBody.h"
#include "Module/ContactsUnion.h"
#include "Module/IterativeConstraintSolver.h"

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"
#include "Collision/CollistionDetectionTriangleSet.h"

namespace dyno
{
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
		this->animationPipeline()->pushModule(cdBV);

		auto merge = std::make_shared<ContactsUnion<TDataType>>();
		elementQuery->outContacts()->connect(merge->inContactsA());
		cdBV->outContacts()->connect(merge->inContactsB());
		this->animationPipeline()->pushModule(merge);

		auto iterSolver = std::make_shared<IterativeConstraintSolver<TDataType>>();
		this->stateTimeStep()->connect(iterSolver->inTimeStep());
		this->varFrictionEnabled()->connect(iterSolver->varFrictionEnabled());
		this->varGravityEnabled()->connect(iterSolver->varGravityEnabled());
		this->varGravityValue()->connect(iterSolver->varGravityValue());
		this->varFrictionCoefficient()->connect(iterSolver->varFrictionCoefficient());
		this->varSlop()->connect(iterSolver->varSlop());
		this->stateMass()->connect(iterSolver->inMass());
		this->stateMass()->connect(merge->inMass());
		this->stateCenter()->connect(iterSolver->inCenter());
		this->stateVelocity()->connect(iterSolver->inVelocity());
		this->stateAngularVelocity()->connect(iterSolver->inAngularVelocity());
		this->stateRotationMatrix()->connect(iterSolver->inRotationMatrix());
		this->stateInertia()->connect(iterSolver->inInertia());
		this->stateQuaternion()->connect(iterSolver->inQuaternion());
		this->stateInitialInertia()->connect(iterSolver->inInitialInertia());

		this->stateBallAndSocketJoints()->connect(iterSolver->inBallAndSocketJoints());
		this->stateSliderJoints()->connect(iterSolver->inSliderJoints());
		this->stateHingeJoints()->connect(iterSolver->inHingeJoints());
		this->stateFixedJoints()->connect(iterSolver->inFixedJoints());
		this->statePointJoints()->connect(iterSolver->inPointJoints());

		this->stateBallAndSocketJoints()->connect(merge->inBallAndSocketJoints());
		this->stateSliderJoints()->connect(merge->inSliderJoints());
		this->stateHingeJoints()->connect(merge->inHingeJoints());
		this->stateFixedJoints()->connect(merge->inFixedJoints());


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
	void Vechicle<TDataType>::updateStates()
	{
		RigidBodySystem<TDataType>::updateStates();

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
}