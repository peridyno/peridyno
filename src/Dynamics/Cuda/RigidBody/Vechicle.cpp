#include "Vechicle.h"

#include "Module/SimpleVechicleDriver.h"
#include "Module/SharedFuncsForRigidBody.h"

namespace dyno
{
	template<typename TDataType>
	Vechicle<TDataType>::Vechicle()
		: RigidBodySystem<TDataType>()
	{
		auto driver = std::make_shared<SimpleVechicleDriver>();

		this->stateFrameNumber()->connect(driver->inFrameNumber());
		this->stateInstanceTransform()->connect(driver->inInstanceTransform());

		this->animationPipeline()->pushModule(driver);
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