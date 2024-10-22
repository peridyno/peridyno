#include "PositionBasedFluidModel.h"

#include "ParticleIntegrator.h"
#include "SummationDensity.h"
#include "IterativeDensitySolver.h"
#include "ImplicitViscosity.h"

#include "Collision/NeighborPointQuery.h"

namespace dyno
{
	IMPLEMENT_TCLASS(PositionBasedFluidModel, TDataType)

	template<typename TDataType>
	PositionBasedFluidModel<TDataType>::PositionBasedFluidModel()
		: GroupModule()
	{
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->inTimeStep()->connect(integrator->inTimeStep());
		this->inPosition()->connect(integrator->inPosition());
		this->inVelocity()->connect(integrator->inVelocity());
		this->inAttribute()->connect(integrator->inAttribute());
		this->pushModule(integrator);

		auto nbrQuery =std::make_shared<NeighborPointQuery<TDataType>>();
		this->varSmoothingLength()->connect(nbrQuery->inRadius());
		this->inPosition()->connect(nbrQuery->inPosition());
		this->pushModule(nbrQuery);

		auto density = std::make_shared<IterativeDensitySolver<TDataType>>();
		this->varSamplingDistance()->connect(density->inSamplingDistance());
		this->varSmoothingLength()->connect(density->inSmoothingLength());
		this->inTimeStep()->connect(density->inTimeStep());
		this->inPosition()->connect(density->inPosition());
		this->inVelocity()->connect(density->inVelocity());
		this->inAttribute()->connect(density->inAttribute());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		this->pushModule(density);
		
		auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
		viscosity->varViscosity()->setValue(Real(1.0));
		this->inTimeStep()->connect(viscosity->inTimeStep());
		this->varSmoothingLength()->connect(viscosity->inSmoothingLength());
		this->inPosition()->connect(viscosity->inPosition());
		this->inVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		this->pushModule(viscosity);
	}

	DEFINE_CLASS(PositionBasedFluidModel);
}