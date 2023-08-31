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
		m_smoothingLength.setValue(Real(0.006));

		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->inTimeStep()->connect(integrator->inTimeStep());
		this->inPosition()->connect(integrator->inPosition());
		this->inVelocity()->connect(integrator->inVelocity());
		this->inForce()->connect(integrator->inForceDensity());
		this->pushModule(integrator);

		auto nbrQuery =std::make_shared<NeighborPointQuery<TDataType>>();
		m_smoothingLength.connect(nbrQuery->inRadius());
		this->inPosition()->connect(nbrQuery->inPosition());
		this->pushModule(nbrQuery);

		auto density = std::make_shared<IterativeDensitySolver<TDataType>>();
		m_smoothingLength.connect(density->inSmoothingLength());
		this->inTimeStep()->connect(density->inTimeStep());
		this->inPosition()->connect(density->inPosition());
		this->inVelocity()->connect(density->inVelocity());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		this->pushModule(density);
		
		auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
		viscosity->varViscosity()->setValue(Real(1.0));
		this->inTimeStep()->connect(viscosity->inTimeStep());
		m_smoothingLength.connect(viscosity->inSmoothingLength());
		this->inPosition()->connect(viscosity->inPosition());
		this->inVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		this->pushModule(viscosity);
	}

	DEFINE_CLASS(PositionBasedFluidModel);
}