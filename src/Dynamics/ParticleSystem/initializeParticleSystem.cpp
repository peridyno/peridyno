#include "initializeParticleSystem.h"

#include "LinearDamping.h"
#include "ParticleIntegrator.h"
#include "ImplicitViscosity.h"
#include "SummationDensity.h"
#include "DensityPBD.h"
#include "BoundaryConstraint.h"
#include "VariationalApproximateProjection.h"

namespace dyno 
{
	ParticleSystemInitializer::ParticleSystemInitializer()
	{
		TypeInfo::New<LinearDamping<DataType3f>>();
		TypeInfo::New<ParticleIntegrator<DataType3f>>();
		TypeInfo::New<ImplicitViscosity<DataType3f>>();
		TypeInfo::New<DensityPBD<DataType3f>>();
		TypeInfo::New<SummationDensity<DataType3f>>();
		TypeInfo::New<VariationalApproximateProjection<DataType3f>>();
		//TypeInfo::New<BoundaryConstraint<DataType3f>>();
	}
}