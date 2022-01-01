#include "IncompressibleModel.h"

#include "ParticleIntegrator.h"
#include "SummationDensity.h"
#include "VelocityConstraint.h"
#include "ImplicitViscosity.h"
#include "Topology/NeighborPointQuery.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(IncompressibleModel, TDataType)

	template<typename TDataType>
	IncompressibleModel<TDataType>::IncompressibleModel()
		: GroupModule()
	{
		m_smoothingLength.setValue(Real(0.006));
		m_samplingDistance.setValue(Real(0.005));

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

		auto density = std::make_shared<VelocityConstraint<TDataType>>();
		m_smoothingLength.connect(density->inSmoothingLength());
		m_samplingDistance.connect(density->inSamplingDistance());
		this->inTimeStep()->connect(density->inTimeStep());
		this->inPosition()->connect(density->inPosition());
		this->inVelocity()->connect(density->inVelocity());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		this->pushModule(density);
		
		auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
		viscosity->varViscosity()->setValue(Real(0.5));
		this->inTimeStep()->connect(viscosity->inTimeStep());
		m_smoothingLength.connect(viscosity->inSmoothingLength());
		this->inPosition()->connect(viscosity->inPosition());
		this->inVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		this->pushModule(viscosity);
	}

	DEFINE_CLASS(IncompressibleModel);
}