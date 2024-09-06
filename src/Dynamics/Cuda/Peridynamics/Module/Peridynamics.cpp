#include "Peridynamics.h"

#include "LinearElasticitySolver.h"

//ParticleSystem
#include "ParticleSystem/Module/ParticleIntegrator.h"

#include "Collision/NeighborPointQuery.h"


namespace dyno
{
	IMPLEMENT_TCLASS(Peridynamics, TDataType)

	template<typename TDataType>
	Peridynamics<TDataType>::Peridynamics()
		: GroupModule()
	{
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->inTimeStep()->connect(integrator->inTimeStep());
		this->inY()->connect(integrator->inPosition());
		this->inVelocity()->connect(integrator->inVelocity());
		this->inForce()->connect(integrator->inForceDensity());
		this->pushModule(integrator);

		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->varHorizon()->connect(nbrQuery->inRadius());
		this->inY()->connect(nbrQuery->inPosition());
		this->pushModule(nbrQuery);

		auto elasticity = std::make_shared<LinearElasticitySolver<TDataType>>();
		this->varHorizon()->connect(elasticity->inHorizon());
		this->inTimeStep()->connect(elasticity->inTimeStep());
		this->inX()->connect(elasticity->inX());
		this->inY()->connect(elasticity->inY());
		this->inVelocity()->connect(elasticity->inVelocity());
		this->inBonds()->connect(elasticity->inBonds());
		this->pushModule(elasticity);
	}

	DEFINE_CLASS(Peridynamics);
}