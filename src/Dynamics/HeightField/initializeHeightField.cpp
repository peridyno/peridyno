#include "initializeHeightField.h"

#include "Ocean.h"
#include "CapillaryWave.h"
#include "OceanPatch.h"

#include "NodeFactory.h"

namespace dyno
{
	std::atomic<HeightFieldInitializer*> HeightFieldInitializer::gInstance;
	std::mutex HeightFieldInitializer::gMutex;

	HeightFieldInitializer::HeightFieldInitializer()
	{
	}

	dyno::PluginEntry* HeightFieldInitializer::instance()
	{
		HeightFieldInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new HeightFieldInitializer();
				ins->setName("Height Field");
				ins->setVersion("1.0");
				ins->setDescription("A height field library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	void HeightFieldInitializer::initializeNodeCreators()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Ocean",
			"ToolBarIco/HeightField/HeightField.png");

		auto group = page->addGroup("Ocean");

		group->addAction(
			"Ocean Patch",
			"ToolBarIco/HeightField/OceanPatch.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<OceanPatch<DataType3f>>(); });

		group->addAction(
			"Ocean",
			"ToolBarIco/HeightField/Ocean.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<Ocean<DataType3f>>(); });

		group->addAction(
			"CapillaryWave",
			"ToolBarIco/HeightField/CapillaryWave.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<CapillaryWave<DataType3f>>(); });
	}

}

dyno::PluginEntry* HeightFieldLibrary::initStaticPlugin()
{
	if (dyno::HeightFieldInitializer::instance()->initialize())
		return dyno::HeightFieldInitializer::instance();

	return nullptr;
}

PERIDYNO_API dyno::PluginEntry* HeightFieldLibrary::initDynoPlugin()
{
	if (dyno::HeightFieldInitializer::instance()->initialize())
		return dyno::HeightFieldInitializer::instance();

	return nullptr;
}
