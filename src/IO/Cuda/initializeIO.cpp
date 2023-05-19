#include "initializeIO.h"

#include "NodeFactory.h"

#include "PointsLoader.h"


namespace dyno
{
	std::atomic<IOInitializer*> IOInitializer::gInstance;
	std::mutex IOInitializer::gMutex;

	PluginEntry* IOInitializer::instance()
	{
		IOInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new IOInitializer();
				ins->setName("Peridynamics");
				ins->setVersion("1.0");
				ins->setDescription("A io library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	IOInitializer::IOInitializer()
		: PluginEntry()
	{
	}

	void IOInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"IO",
			"ToolBarIco/Interaction/Interaction.png");

		auto group = page->addGroup("Interaction");

		group->addAction(
			"Points Loader",
			"ToolBarIco/Interaction/PointsLoader_v4.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<PointsLoader<DataType3f>>();
			});
	}
}

dyno::PluginEntry* IO::initStaticPlugin()
{
	if (dyno::IOInitializer::instance()->initialize())
		return dyno::IOInitializer::instance();

	return nullptr;
}