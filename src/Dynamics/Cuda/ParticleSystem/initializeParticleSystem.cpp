#include "initializeParticleSystem.h"

#include "Module/LinearDamping.h"
#include "Module/ParticleIntegrator.h"
#include "Module/ImplicitViscosity.h"
#include "Module/SummationDensity.h"
#include "Module/IterativeDensitySolver.h"
#include "Module/BoundaryConstraint.h"
#include "Module/VariationalApproximateProjection.h"

#include "Emitters/CircularEmitter.h"
#include "Emitters/SquareEmitter.h"
#include "Emitters/PoissonEmitter.h"

#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"

#include "Module/CalculateNorm.h"

#include "ColorMapping.h"

#include "ParticleFluid.h"

#include "NodeFactory.h"

namespace dyno 
{
	std::atomic<ParticleSystemInitializer*> ParticleSystemInitializer::gInstance;
	std::mutex ParticleSystemInitializer::gMutex;

	PluginEntry* ParticleSystemInitializer::instance()
	{
		ParticleSystemInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new ParticleSystemInitializer();
				ins->setName("Particle System");
				ins->setVersion("1.0");
				ins->setDescription("A particle system library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	ParticleSystemInitializer::ParticleSystemInitializer()
		: PluginEntry()
	{
	}

	void ParticleSystemInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Particle System", 
			"ToolBarIco/ParticleSystem/ParticleSystem.png");

		auto emitters = page->addGroup("Emitters");

		emitters->addAction(
			"Circular Emitter",
			"ToolBarIco/ParticleSystem/ParticleEmitterRound.png",
			[=]()->std::shared_ptr<Node> {
				auto emitter = std::make_shared<CircularEmitter<DataType3f>>();
				return emitter;
			});

		emitters->addAction(
			"Square Emitter",
			"ToolBarIco/ParticleSystem/ParticleEmitterSquare.png",
			[=]()->std::shared_ptr<Node> {
				auto emitter = std::make_shared<SquareEmitter<DataType3f>>();
				return emitter;;
			});

		emitters->addAction(
			"Poisson Emitter",
			//"ToolBarIco/ParticleSystem/PoissonEmitter.png",
			"ToolBarIco/ParticleSystem/ParticleEmitterSquare.png",
			[=]()->std::shared_ptr<Node> {
				auto emitter = std::make_shared<PoissonEmitter<DataType3f>>();
				return emitter;
			});

		auto solvers = page->addGroup("Particle Fluid Solvers");

		solvers->addAction(
			"Particle Fluid",
			"ToolBarIco/ParticleSystem/ParticleFluid.png",
			[=]()->std::shared_ptr<Node> { 
				auto fluid = std::make_shared<ParticleFluid<DataType3f>>();
				return fluid; 
			});
	}
}

PERIDYNO_API dyno::PluginEntry* PaticleSystem::initDynoPlugin()
{
	if (dyno::ParticleSystemInitializer::instance()->initialize())
		return dyno::ParticleSystemInitializer::instance();

	return nullptr;
}

dyno::PluginEntry* PaticleSystem::initStaticPlugin()
{
	if (dyno::ParticleSystemInitializer::instance()->initialize())
		return dyno::ParticleSystemInitializer::instance();

	return nullptr;
}
