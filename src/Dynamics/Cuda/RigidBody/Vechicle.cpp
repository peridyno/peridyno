#include "Vechicle.h"

#include "Module/SimpleVechicleDriver.h"
#include "Module/SharedFuncsForRigidBody.h"

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"
#include "Collision/CollistionDetectionTriangleSet.h"

#include "ContactsUnion.h"
#include "IterativeConstraintSolver.h"

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

		this->stateBinding()->resize(sizeOfRigids);
		this->stateBindingTag()->resize(sizeOfRigids);

		auto texMesh = this->inTextureMesh()->constDataPtr();

		uint N = texMesh->shapes().size();

		CArrayList<Transform3f> tms;
		tms.resize(N, 1);

		for (uint i = 0; i < N; i++)
		{
			tms[i].insert(texMesh->shapes()[i]->boundingTransform);
		}

		if (this->stateInstanceTransform()->isEmpty())
		{
			this->stateInstanceTransform()->allocate();
		}

		auto instantanceTransform = this->stateInstanceTransform()->getDataPtr();
		instantanceTransform->assign(tms);

		tms.clear();

		auto binding = this->stateBinding()->getDataPtr();
		auto bindingtag = this->stateBindingTag()->getDataPtr();


		std::vector<Pair<uint, uint>> bindingPair(sizeOfRigids);
		std::vector<int> tags(sizeOfRigids, 0);

		for (int i = 0; i < mBindingPair.size(); i++)
		{
			bindingPair[mBodyId[i]] = mBindingPair[i];
			tags[mBodyId[i]] = 1;
		}

		binding->assign(bindingPair);
		bindingtag->assign(tags);

		mInitialRot.assign(this->stateRotationMatrix()->constData());

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
			this->stateBinding()->getData(),
			this->stateBindingTag()->getData());
	}

	template<typename TDataType>
	void Vechicle<TDataType>::bind(uint bodyId, Pair<uint, uint> shapeId)
	{
		mBindingPair.push_back(shapeId);
		mBodyId.push_back(bodyId);
	}

	DEFINE_CLASS(Vechicle);
}