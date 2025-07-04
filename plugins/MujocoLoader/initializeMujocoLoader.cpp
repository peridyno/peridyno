#include "initializeMujocoLoader.h"
#include "NodeFactory.h"
#include "MujocoXMLLoader.h"

namespace dyno
{
	std::atomic<MujocoInitializer*> MujocoInitializer::gInstance;
	std::mutex MujocoInitializer::gMutex;

	PluginEntry* MujocoInitializer::instance()
	{
		MujocoInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new MujocoInitializer();
				ins->setName("MujocoLoader");
				ins->setVersion("1.0");
				ins->setDescription("A Mujoco library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	void MujocoInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		factory->addContentAction(std::string("xml"),
			[=](const std::string& path)->std::shared_ptr<Node>
			{
				auto node = std::make_shared<MujocoXMLLoader<DataType3f>>();
				node->varFilePath()->setValue(path);
				return node;
			});

		auto page = factory->addPage(
			"IO",
			"ToolBarIco/Modeling/Modeling.png");

		auto group = page->addGroup("Modeling");

		group->addAction(
			"Mujoco Loader",
			"ToolBarIco/Modeling/TriangularMesh.png",
			[=]()->std::shared_ptr<Node> {
				auto node = std::make_shared<MujocoXMLLoader<DataType3f>>();

				return node;
			});

	}
}

dyno::PluginEntry* MujocoLoader::initStaticPlugin()
{
	if (dyno::MujocoInitializer::instance()->initialize())
		return dyno::MujocoInitializer::instance();

	return nullptr;
}

PERIDYNO_API dyno::PluginEntry* MujocoLoader::initDynoPlugin()
{
	if (dyno::MujocoInitializer::instance()->initialize())
		return dyno::MujocoInitializer::instance();

	return nullptr;
}