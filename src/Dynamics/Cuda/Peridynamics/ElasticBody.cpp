#include "ElasticBody.h"

#include "Module/ProjectivePeridynamics.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ElasticBody, TDataType)

	template<typename TDataType>
	ElasticBody<TDataType>::ElasticBody()
		: Peridynamics<TDataType>()
	{
		auto peri = std::make_shared<ProjectivePeridynamics<TDataType>>();
		this->stateTimeStep()->connect(peri->inTimeStep());
		this->stateHorizon()->connect(peri->inHorizon());
		this->stateReferencePosition()->connect(peri->inX());
		this->statePosition()->connect(peri->inY());
		this->stateVelocity()->connect(peri->inVelocity());
		this->stateBonds()->connect(peri->inBonds());
		this->animationPipeline()->pushModule(peri);
	}

	template<typename TDataType>
	ElasticBody<TDataType>::~ElasticBody()
	{
		
	}

	DEFINE_CLASS(ElasticBody);
}