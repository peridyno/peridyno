#include "initializeInteraction.h"

#include "NodeFactory.h"

#include "PickerNode.h"

namespace dyno 
{
	std::atomic<InteractionInitializer*> InteractionInitializer::gInstance;
	std::mutex InteractionInitializer::gMutex;

	PluginEntry* InteractionInitializer::instance()
	{
		InteractionInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new InteractionInitializer();
				ins->setName("Interaction");
				ins->setVersion("1.0");
				ins->setDescription("A interaction library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	void InteractionInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Interaction", 
			"ToolBarIco/Interaction/Interaction.png");

		auto group = page->addGroup("Interaction");

		group->addAction(
			"Picker",
			"ToolBarIco/Interaction/Picker.png",//48px-Image-x-generic.png

		[=]()->std::shared_ptr<Node> { return std::make_shared<PickerNode<DataType3f>>(); });
	}
}
//

dyno::PluginEntry* Interaction::initStaticPlugin()
{
	if (dyno::InteractionInitializer::instance()->initialize())
		return dyno::InteractionInitializer::instance();

	return nullptr;
}

PERIDYNO_API dyno::PluginEntry* Interaction::initDynoPlugin()
{
	if (dyno::InteractionInitializer::instance()->initialize())
		return dyno::InteractionInitializer::instance();

	return nullptr;
}
