#include "initializeMultiphysics.h"

#include "NodeFactory.h"

#include "VolumeBoundary.h"
#include "AdaptiveBoundary.h"

namespace dyno
{
	std::atomic<MultiphysicsInitializer*> MultiphysicsInitializer::gInstance;
	std::mutex MultiphysicsInitializer::gMutex;

	PluginEntry* MultiphysicsInitializer::instance()
	{
		MultiphysicsInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new MultiphysicsInitializer();
				ins->setName("Peridynamics");
				ins->setVersion("1.0");
				ins->setDescription("A multiphysics library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	MultiphysicsInitializer::MultiphysicsInitializer()
		: PluginEntry()
	{
	}

	void MultiphysicsInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Volume",
			"ToolBarIco/Volume/GenerateSparseVolume.png");

		auto group = page->addGroup("Volume");

		group->addAction(
			"Adaptive Boundary",
			"ToolBarIco/Volume/AdaptiveBoundary.png",
			[=]()->std::shared_ptr<Node> {
				auto node = std::make_shared<AdaptiveBoundary<DataType3f>>();
				return node;
			});
	}
}

PERIDYNO_API dyno::PluginEntry* Multiphysics::initDynoPlugin()
{
	if (dyno::MultiphysicsInitializer::instance()->initialize())
		return dyno::MultiphysicsInitializer::instance();

	return nullptr;
}

dyno::PluginEntry* Multiphysics::initStaticPlugin()
{
	if (dyno::MultiphysicsInitializer::instance()->initialize())
		return dyno::MultiphysicsInitializer::instance();

	return nullptr;
}
