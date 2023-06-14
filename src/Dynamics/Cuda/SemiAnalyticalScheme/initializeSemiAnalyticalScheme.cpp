#include "initializeSemiAnalyticalScheme.h"

#include "NodeFactory.h"

#include "ParticleSystem/ParticleSystem.h"

#include "SemiAnalyticalSFINode.h"

#include "ComputeParticleAnisotropy.h"
#include "SemiAnalyticalSummationDensity.h"
#include "SemiAnalyticalPBD.h"

namespace dyno 
{
	std::atomic<SemiAnalyticalSchemeInitializer*> SemiAnalyticalSchemeInitializer::gInstance;
	std::mutex SemiAnalyticalSchemeInitializer::gMutex;

	dyno::PluginEntry* SemiAnalyticalSchemeInitializer::instance()
	{
		SemiAnalyticalSchemeInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new SemiAnalyticalSchemeInitializer();
				ins->setName("SemiAnalyticalScheme");
				ins->setVersion("1.0");
				ins->setDescription("A semi-analytical scheme library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	void SemiAnalyticalSchemeInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Particle System", 
			"ToolBarIco/ParticleSystem/ParticleSystem.png");

		auto group = page->addGroup("Semi Analytical Scheme");

		group->addAction(
			"Semi Analytical SFI",
			"ToolBarIco/ParticleSystem/SemiAnalvticalSFI_yellow.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<SemiAnalyticalSFINode<DataType3f>>(); });


	}

	dyno::PluginEntry* SemiAnalyticalScheme::initStaticPlugin()
	{
		if (dyno::SemiAnalyticalSchemeInitializer::instance()->initialize())
			return dyno::SemiAnalyticalSchemeInitializer::instance();

		return nullptr;
	}

	PERIDYNO_API dyno::PluginEntry* SemiAnalyticalScheme::initDynoPlugin()
	{
		if (dyno::SemiAnalyticalSchemeInitializer::instance()->initialize())
			return dyno::SemiAnalyticalSchemeInitializer::instance();

		return nullptr;
	}

}