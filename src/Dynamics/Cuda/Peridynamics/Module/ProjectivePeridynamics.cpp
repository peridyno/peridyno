#include "ProjectivePeridynamics.h"

#include "LinearElasticitySolver.h"

//ParticleSystem
#include "ParticleSystem/Module/ParticleIntegrator.h"

#include "Collision/NeighborPointQuery.h"


namespace dyno
{
	IMPLEMENT_TCLASS(ProjectivePeridynamics, TDataType)

	template<typename TDataType>
	ProjectivePeridynamics<TDataType>::ProjectivePeridynamics()
		: GroupModule()
	{
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->inTimeStep()->connect(integrator->inTimeStep());
		this->inY()->connect(integrator->inPosition());
		this->inVelocity()->connect(integrator->inVelocity());
		this->pushModule(integrator);

		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->inHorizon()->connect(nbrQuery->inRadius());
		this->inY()->connect(nbrQuery->inPosition());
		this->pushModule(nbrQuery);

		auto elasticity = std::make_shared<LinearElasticitySolver<TDataType>>();
		this->inTimeStep()->connect(elasticity->inTimeStep());
		this->inHorizon()->connect(elasticity->inHorizon());
		this->inX()->connect(elasticity->inX());
		this->inY()->connect(elasticity->inY());
		this->inVelocity()->connect(elasticity->inVelocity());
		this->inBonds()->connect(elasticity->inBonds());
		this->pushModule(elasticity);
	}

	DEFINE_CLASS(ProjectivePeridynamics);
}