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

#include "PoissonDiskSampling.h"

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

		auto group = page->addGroup("Particle System");

		group->addAction(
			"Circular Emitter",
			"ToolBarIco/ParticleSystem/ParticleEmitterRound.png",
			[=]()->std::shared_ptr<Node> {
				auto emitter = std::make_shared<CircularEmitter<DataType3f>>();

				auto wireRender = std::make_shared<GLWireframeVisualModule>();
				wireRender->setColor(Color(0, 1, 0));
				emitter->stateOutline()->connect(wireRender->inEdgeSet());
				emitter->graphicsPipeline()->pushModule(wireRender);
				return emitter;
			});

		group->addAction(
			"Square Emitter",
			"ToolBarIco/ParticleSystem/ParticleEmitterSquare.png",
			[=]()->std::shared_ptr<Node> {
				auto emitter = std::make_shared<SquareEmitter<DataType3f>>();

				auto wireRender = std::make_shared<GLWireframeVisualModule>();
				wireRender->setColor(Color(0, 1, 0));
				emitter->stateOutline()->connect(wireRender->inEdgeSet());
				emitter->graphicsPipeline()->pushModule(wireRender);
				return emitter;;
			});

		group->addAction(
			"Poisson Emitter",
			//"ToolBarIco/ParticleSystem/PoissonEmitter.png",
			"ToolBarIco/ParticleSystem/ParticleEmitterSquare.png",
			[=]()->std::shared_ptr<Node> {
				auto emitter = std::make_shared<PoissonEmitter<DataType3f>>();

				auto wireRender = std::make_shared<GLWireframeVisualModule>();
				wireRender->setColor(Color(0, 1, 0));
				emitter->stateOutline()->connect(wireRender->inEdgeSet());
				emitter->graphicsPipeline()->pushModule(wireRender);
				return emitter;
			});

		group->addAction(
			"Particle Fluid",
			"ToolBarIco/ParticleSystem/ParticleFluid.png",
			[=]()->std::shared_ptr<Node> { 
				auto fluid = std::make_shared<ParticleFluid<DataType3f>>();

				auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
				fluid->stateVelocity()->connect(calculateNorm->inVec());
				fluid->graphicsPipeline()->pushModule(calculateNorm);

				auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
				colorMapper->varMax()->setValue(5.0f);
				calculateNorm->outNorm()->connect(colorMapper->inScalar());
				fluid->graphicsPipeline()->pushModule(colorMapper);

				auto ptRender = std::make_shared<GLPointVisualModule>();
				ptRender->varBaseColor()->setValue(Color(1, 0, 0));
				ptRender->varPointSize()->setValue(0.002f);
				ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

				fluid->statePointSet()->connect(ptRender->inPointSet());
				colorMapper->outColor()->connect(ptRender->inColor());

				fluid->graphicsPipeline()->pushModule(ptRender);

				return fluid; 
			});

		group->addAction(
			"Poisson Disk Sampler",
			"ToolBarIco/Modeling/PoissonDiskSampler_v2.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<PoissonDiskSampling<DataType3f>>();
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
