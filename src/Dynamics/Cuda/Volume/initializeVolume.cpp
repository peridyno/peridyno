#include "initializeVolume.h"

#include "NodeFactory.h"

#include "VolumeGenerator.h"
#include "VolumeBoolean.h"
#include "MarchingCubes.h"

namespace dyno
{
	std::atomic<VolumeInitializer*> VolumeInitializer::gInstance;
	std::mutex VolumeInitializer::gMutex;

	PluginEntry* VolumeInitializer::instance()
	{
		VolumeInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new VolumeInitializer();
				ins->setName("Particle System");
				ins->setVersion("1.0");
				ins->setDescription("A particle system library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	VolumeInitializer::VolumeInitializer()
		: PluginEntry()
	{
	}

	void VolumeInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Volume",
			"ToolBarIco/Volume/GenerateSparseVolume.png");

		auto group = page->addGroup("Volume");

		group->addAction(
			"Volume",
			"ToolBarIco/Volume/volume_v3.png",
			[=]()->std::shared_ptr<Node> {
				auto node = std::make_shared<VolumeGenerator<DataType3f>>();

				return node;
			});

		group->addAction(
			"VolumeBoolean",
			"ToolBarIco/Volume/Intersect_v5.png",
			[=]()->std::shared_ptr<Node> {
				auto node = std::make_shared<VolumeBoolean<DataType3f>>();

				return node;
			});

		group->addAction(
			"MarchingCubes",
			"ToolBarIco/Volume/GenerateUniformVolume.png",
			[=]()->std::shared_ptr<Node> {
				auto node = std::make_shared<MarchingCubes<DataType3f>>();

				return node; }
		);
	}
}

dyno::PluginEntry* Volume::initStaticPlugin()
{
	if (dyno::VolumeInitializer::instance()->initialize())
		return dyno::VolumeInitializer::instance();

	return nullptr;
}

PERIDYNO_API dyno::PluginEntry* Volume::initDynoPlugin()
{
	if (dyno::VolumeInitializer::instance()->initialize())
		return dyno::VolumeInitializer::instance();

	return nullptr;
}
