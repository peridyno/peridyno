#include "initializeDualParticleSystem.h"
#include "DualParticleFluid.h"
#include "Module/DualParticleIsphModule.h"
#include "Module/VirtualColocationStrategy.h"
#include "Module/VirtualParticleShiftingStrategy.h"
#include "Module/VirtualSpatiallyAdaptiveStrategy.h"
#include "ParticleSystem/Emitters/CircularEmitter.h"
#include "ParticleSystem/Emitters/SquareEmitter.h"
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

		auto group = page->addGroup("Particle Fluid Solvers");


		group->addAction(
			"Dual Particle Fluid",
			"ToolBarIco/DualParticleSystem/DualParticleFluid_v4.png",
			[=]()->std::shared_ptr<Node> { 
				
				auto fluid = std::make_shared<DualParticleFluid<DataType3f>>();

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
