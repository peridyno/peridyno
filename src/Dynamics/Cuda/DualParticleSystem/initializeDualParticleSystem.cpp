#include "initializeDualParticleSystem.h"
#include "DualParticleFluidSystem.h"
#include "DualParticleIsphModule.h"
#include "VirtualColocationStrategy.h"
#include "VirtualParticleShiftingStrategy.h"
#include "VirtualSpatiallyAdaptiveStrategy.h"
#include "ParticleSystem/CircularEmitter.h"
#include "ParticleSystem/SquareEmitter.h"
#include "ParticleSystem/ParticleFluid.h"
#include "GLPointVisualModule.h"
#include "ColorMapping.h"
#include "Module/CalculateNorm.h"

#include "NodeFactory.h"

namespace dyno 
{
	std::atomic<DualParticleSystemInitializer*> DualParticleSystemInitializer::gInstance;
	std::mutex DualParticleSystemInitializer::gMutex;

	PluginEntry* DualParticleSystemInitializer::instance()
	{
		DualParticleSystemInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new DualParticleSystemInitializer();
				ins->setName("Dual Particle System");
				ins->setVersion("1.0");
				ins->setDescription("A dual particle system library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	DualParticleSystemInitializer::DualParticleSystemInitializer()
		: PluginEntry()
	{
	}

	void DualParticleSystemInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Particle System",
			"ToolBarIco/DualParticleSystem/DualParticleSystem_v4.png");

		auto group = page->addGroup("Dual Particle System");


		group->addAction(
			"Dual Particle Fluid",
			"ToolBarIco/DualParticleSystem/DualParticleFluid_v4.png",
			[=]()->std::shared_ptr<Node> { 
				
				auto fluid = std::make_shared<DualParticleFluidSystem<DataType3f>>();

				auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
				fluid->stateVelocity()->connect(calculateNorm->inVec());
				fluid->graphicsPipeline()->pushModule(calculateNorm);

				auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
				colorMapper->varMax()->setValue(5.0f);
				calculateNorm->outNorm()->connect(colorMapper->inScalar());
				fluid->graphicsPipeline()->pushModule(colorMapper);

				auto ptRender = std::make_shared<GLPointVisualModule>();
				ptRender->setColor(Color(1, 0, 0));
				ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

				fluid->statePointSet()->connect(ptRender->inPointSet());
				colorMapper->outColor()->connect(ptRender->inColor());
				fluid->graphicsPipeline()->pushModule(ptRender);

				auto vpRender = std::make_shared<GLPointVisualModule>();
				vpRender->setColor(Color(1, 1, 0));
				vpRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
				fluid->stateVirtualPointSet()->connect(vpRender->inPointSet());
				vpRender->varPointSize()->setValue(0.0005);
				fluid->graphicsPipeline()->pushModule(vpRender);

				return fluid; 
			});


	
	}

}

PERIDYNO_API dyno::PluginEntry* DualParticleSystem::initDynoPlugin()
{
	if (dyno::DualParticleSystemInitializer::instance()->initialize())
		return dyno::DualParticleSystemInitializer::instance();

	return nullptr;
}

dyno::PluginEntry* DualParticleSystem::initStaticPlugin()
{
	if (dyno::DualParticleSystemInitializer::instance()->initialize())
		return dyno::DualParticleSystemInitializer::instance();

	return nullptr;
}
