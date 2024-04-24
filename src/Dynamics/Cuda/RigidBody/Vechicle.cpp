#include "Vechicle.h"

#include "Module/SimpleVechicleDriver.h"

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
	}

	template<typename TDataType>
	void Vechicle<TDataType>::updateStates()
	{
		RigidBodySystem<TDataType>::updateStates();
	}

	DEFINE_CLASS(Vechicle);
}