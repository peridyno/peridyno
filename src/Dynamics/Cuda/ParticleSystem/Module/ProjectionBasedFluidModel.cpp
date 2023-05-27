#include "ProjectionBasedFluidModel.h"

#include "VariationalApproximateProjection.h"
#include "ParticleIntegrator.h"
#include "SummationDensity.h"

#include "ImplicitViscosity.h"
#include "Collision/NeighborPointQuery.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ProjectionBasedFluidModel, TDataType)

	template<typename TDataType>
	ProjectionBasedFluidModel<TDataType>::ProjectionBasedFluidModel()
		: GroupModule()
	{
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->inTimeStep()->connect(integrator->inTimeStep());
		this->inPosition()->connect(integrator->inPosition());
		this->inVelocity()->connect(integrator->inVelocity());
		this->inForce()->connect(integrator->inForceDensity());
		this->inAttribute()->connect(integrator->inAttribute());
		this->pushModule(integrator);

		auto nbrQuery =std::make_shared<NeighborPointQuery<TDataType>>();
		this->varSmoothingLength()->connect(nbrQuery->inRadius());
		this->inPosition()->connect(nbrQuery->inPosition());
		this->pushModule(nbrQuery);

		auto density = std::make_shared<VariationalApproximateProjection<TDataType>>();
		this->varSmoothingLength()->connect(density->inSmoothingLength());
		this->varSamplingDistance()->connect(density->inSamplingDistance());
		this->inTimeStep()->connect(density->inTimeStep());
		this->inPosition()->connect(density->inPosition());
		this->inVelocity()->connect(density->inVelocity());
		this->inNormal()->connect(density->inNormal());
		this->inAttribute()->connect(density->inAttribute());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		this->pushModule(density);
		
		auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
		viscosity->varViscosity()->setValue(Real(0.5));
		this->inTimeStep()->connect(viscosity->inTimeStep());
		this->varSmoothingLength()->connect(viscosity->inSmoothingLength());
		this->inPosition()->connect(viscosity->inPosition());
		this->inVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		this->pushModule(viscosity);
	}

	DEFINE_CLASS(ProjectionBasedFluidModel);
}