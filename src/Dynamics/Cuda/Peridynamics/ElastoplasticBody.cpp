#include "ElastoplasticBody.h"

#include "Collision/NeighborPointQuery.h"

#include "Module/ElastoplasticityModule.h"

#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ElastoplasticBody, TDataType)

	template<typename TDataType>
	ElastoplasticBody<TDataType>::ElastoplasticBody()
		: Peridynamics<TDataType>()
	{
		auto m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->stateTimeStep()->connect(m_integrator->inTimeStep());
		this->statePosition()->connect(m_integrator->inPosition());
		this->stateVelocity()->connect(m_integrator->inVelocity());
		this->animationPipeline()->pushModule(m_integrator);
		
		auto m_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->stateHorizon()->connect(m_nbrQuery->inRadius());
		this->statePosition()->connect(m_nbrQuery->inPosition());
		this->animationPipeline()->pushModule(m_nbrQuery);

		auto m_plasticity = std::make_shared<ElastoplasticityModule<TDataType>>();
		this->stateHorizon()->connect(m_plasticity->inHorizon());
		this->stateTimeStep()->connect(m_plasticity->inTimeStep());
		this->statePosition()->connect(m_plasticity->inY());
		this->stateReferencePosition()->connect(m_plasticity->inX());
		this->stateVelocity()->connect(m_plasticity->inVelocity());
		this->stateBonds()->connect(m_plasticity->inBonds());
		m_nbrQuery->outNeighborIds()->connect(m_plasticity->inNeighborIds());
		this->animationPipeline()->pushModule(m_plasticity);

		auto m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
		m_visModule->varViscosity()->setValue(Real(1));
		this->stateTimeStep()->connect(m_visModule->inTimeStep());
		this->stateHorizon()->connect(m_visModule->inSmoothingLength());
		this->statePosition()->connect(m_visModule->inPosition());
		this->stateVelocity()->connect(m_visModule->inVelocity());
		m_nbrQuery->outNeighborIds()->connect(m_visModule->inNeighborIds());
		this->animationPipeline()->pushModule(m_visModule);
	}

	template<typename TDataType>
	ElastoplasticBody<TDataType>::~ElastoplasticBody()
	{
		
	}

	DEFINE_CLASS(ElastoplasticBody);
}