#include "Peridynamics.h"

#include "ParticleSystem/ParticleIntegrator.h"
#include "Topology/NeighborPointQuery.h"
#include "ElasticityModule.h"

namespace dyno
{
	IMPLEMENT_TCLASS(Peridynamics, TDataType)

	template<typename TDataType>
	Peridynamics<TDataType>::Peridynamics()
		: GroupModule()
	{
		auto m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->inTimeStep()->connect(m_integrator->inTimeStep());
		this->inPosition()->connect(m_integrator->inPosition());
		this->inVelocity()->connect(m_integrator->inVelocity());
		this->inForce()->connect(m_integrator->inForceDensity());
		this->pushModule(m_integrator);

		auto m_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->varHorizon()->connect(m_nbrQuery->inRadius());
		this->inPosition()->connect(m_nbrQuery->inPosition());
		this->pushModule(m_nbrQuery);

		auto m_elasticity = std::make_shared<ElasticityModule<TDataType>>();
		this->varHorizon()->connect(m_elasticity->inHorizon());
		this->inTimeStep()->connect(m_elasticity->inTimeStep());
		this->inPosition()->connect(m_elasticity->inPosition());
		this->inVelocity()->connect(m_elasticity->inVelocity());
		this->inRestShape()->connect(m_elasticity->inRestShape());
		m_nbrQuery->outNeighborIds()->connect(m_elasticity->inNeighborIds());
		this->pushModule(m_elasticity);
	}

	DEFINE_CLASS(Peridynamics);
}