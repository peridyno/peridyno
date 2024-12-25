#include "InstanceTransform.h"

#include "SharedFuncsForRigidBody.h"

namespace dyno
{
	IMPLEMENT_TCLASS(InstanceTransform, TDataType);

	template<typename TDataType>
	InstanceTransform<TDataType>::InstanceTransform()
		: ComputeModule()
	{

	}

	template<typename TDataType>
	InstanceTransform<TDataType>::~InstanceTransform()
	{
	}


	template<typename TDataType>
	void InstanceTransform<TDataType>::compute()
	{
		this->outInstanceTransform()->assign(this->inInstanceTransform()->constData());

		ApplyTransform(
			this->outInstanceTransform()->getData(),
			this->inCenter()->getData(),
			this->inRotationMatrix()->getData(),
			this->inInitialRotation()->constData(),
			this->inBindingPair()->constData(),
			this->inBindingTag()->constData());
	}

	DEFINE_CLASS(InstanceTransform);
}