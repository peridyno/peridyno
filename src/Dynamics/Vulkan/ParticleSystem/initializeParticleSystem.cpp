#include "initializeParticleSystem.h"

#include "SquareEmitter.h"

#include "ParticleFluid.h"

#include "NodeFactory.h"

#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"

#include "Module/CalculateNorm.h"
#include "ColorMapping.h"

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
			"Square Emitter",
			"ToolBarIco/ParticleSystem/ParticleEmitterSquare.png",
			[=]()->std::shared_ptr<Node> {
				auto emitter = std::make_shared<SquareEmitter>();

				auto wireRender = std::make_shared<GLWireframeVisualModule>();
				wireRender->setColor(Color(0, 1, 0));
				emitter->stateOutline()->connect(wireRender->inEdgeSet());
				emitter->graphicsPipeline()->pushModule(wireRender);
				return emitter;;
			});

		group->addAction(
			"Particle Fluid",
			"ToolBarIco/ParticleSystem/ParticleFluid.png",
			[=]()->std::shared_ptr<Node> { 
				auto fluid = std::make_shared<ParticleFluid>();

				auto calculateNorm = std::make_shared<CalculateNorm>();
				fluid->stateVelocity()->connect(calculateNorm->inVec());
				fluid->graphicsPipeline()->pushModule(calculateNorm);

				auto colorMapper = std::make_shared<ColorMapping>();
				colorMapper->varMax()->setValue(5.0f);
				calculateNorm->outNorm()->connect(colorMapper->inScalar());
				fluid->graphicsPipeline()->pushModule(colorMapper);

				auto pointRender = std::make_shared<GLPointVisualModule>();
				pointRender->varColorMode()->setCurrentKey(GLPointVisualModule::PER_VERTEX_SHADER);
				fluid->statePointSet()->connect(pointRender->inPointSet());
				colorMapper->outColor()->connect(pointRender->inColor());

				fluid->graphicsPipeline()->pushModule(pointRender);

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
