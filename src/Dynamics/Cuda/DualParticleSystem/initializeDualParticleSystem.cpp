#include "initializeDualParticleSystem.h"
#include "DualParticleFluidSystem.h"
#include "DualParticleIsphModule.h"
#include "ParticleMeshCollidingNode.h"
#include "VirtualColocationStrategy.h"
#include "VirtualParticleShiftingStrategy.h"
#include "VirtualSpatiallyAdaptiveStrategy.h"
#include "ParticleSystem/CircularEmitter.h"
#include "ParticleSystem/SquareEmitter.h"
#include "ParticleSystem/ParticleFluid.h"


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
			"Dual Particle System", 
			"ToolBarIco/DualParticleSystem/DualParticleSystem_v4.png");

		auto group = page->addGroup("Dual Particle System");

// 		group->addAction(
// 			"Round Particle Emitter", 
// 			"ToolBarIco/ParticleSystem/ParticleEmitterRound.png",
// 			[=]()->std::shared_ptr<Node> { return std::make_shared<CircularEmitter<DataType3f>>(); });
// 
// 		group->addAction(
// 			"Square Particle Emitter",
// 			"ToolBarIco/ParticleSystem/ParticleEmitterSquare.png",
// 			[=]()->std::shared_ptr<Node> { return std::make_shared<SquareEmitter<DataType3f>>(); });

		group->addAction(
			"Dual Particle Fluid",
			"ToolBarIco/DualParticleSystem/DualParticleFluid_v4.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<DualParticleFluidSystem<DataType3f>>(); });

		//group->addAction(
		//	"SIMPLE Iteration Particle Fluid",
		//	"ToolBarIco/DualParticleSystem/SIMPLEIterationParticleFluid.png",
		//	[=]()->std::shared_ptr<Node> { return std::make_shared<DualParticleFluid<DataType3f>>(); });
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
