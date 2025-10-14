#include "initializeSkeletonLoader.h"

#include "NodeFactory.h"

#include "FBXLoader.h"
#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"
#include "GLPointVisualModule.h"


namespace dyno
{
	std::atomic<FBXInitializer*> FBXInitializer::gInstance;
	std::mutex FBXInitializer::gMutex;

	PluginEntry* FBXInitializer::instance()
	{
		FBXInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new FBXInitializer();
				ins->setName("FBX Loader");
				ins->setVersion("1.0");
				ins->setDescription("A FBX Loader library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	void FBXInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		factory->addContentAction(std::string("fbx"),
			[=](const std::string& path)->std::shared_ptr<Node>
			{
				auto node = std::make_shared<FBXLoader<DataType3f>>();
				node->varFileName()->setValue(path);
				return node;
			});

		auto page = factory->addPage(
			"IO",
			"ToolBarIco/Modeling/Modeling.png");

		auto group = page->addGroup("Modeling");

		group->addAction(
			"FBX Loader",
			"ToolBarIco/Modeling/TriangularMesh.png",
			[=]()->std::shared_ptr<Node> {
				auto node = std::make_shared<FBXLoader<DataType3f>>();

				return node;
			});




	}
}

dyno::PluginEntry* FBXLoader::initStaticPlugin()
{
	if (dyno::FBXInitializer::instance()->initialize())
		return dyno::FBXInitializer::instance();

	return nullptr;
}

PERIDYNO_API dyno::PluginEntry* FBXLoader::initDynoPlugin()
{
	if (dyno::FBXInitializer::instance()->initialize())
		return dyno::FBXInitializer::instance();

	return nullptr;
}

