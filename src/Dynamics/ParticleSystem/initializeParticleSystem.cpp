#include "initializeParticleSystem.h"

#include "LinearDamping.h"
#include "ParticleIntegrator.h"
#include "ImplicitViscosity.h"
#include "SummationDensity.h"
#include "DensityPBD.h"
#include "BoundaryConstraint.h"
#include "VariationalApproximateProjection.h"

#include "ParticleEmitterRound.h"
#include "ParticleEmitterSquare.h"

#include "NodeFactory.h"

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


		initializeNodeCreators();
	}

	void ParticleSystemInitializer::initializeNodeCreators()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto group = factory->addGroup(
			"Particle System", 
			"Particle System", 
			"ToolBarIco/ParticleSystem/ParticleSystem.png");

		group->addAction(
			"Particle Emitter 1", 
			"ToolBarIco/ParticleSystem/ParticleEmitterRound.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<ParticleEmitterRound<DataType3f>>(); });

		group->addAction(
			"Particle Emitter 2",
			"ToolBarIco/ParticleSystem/ParticleEmitterSquare.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<ParticleEmitterSquare<DataType3f>>(); });

		group->addAction(
			"Particle Fluid",
			"ToolBarIco/ParticleSystem/ParticleFluid.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<ParticleSystem<DataType3f>>(); });
	}

}