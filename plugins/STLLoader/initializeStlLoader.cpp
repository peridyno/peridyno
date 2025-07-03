#include "initializeStlLoader.h"

#include "NodeFactory.h"

#include "StlLoader.h"
#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"
#include "GLPointVisualModule.h"


namespace dyno 
{
	std::atomic<STLInitializer*> STLInitializer::gInstance;
	std::mutex STLInitializer::gMutex;

	PluginEntry* STLInitializer::instance()
	{
		STLInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new STLInitializer();
				ins->setName("StlLoader");
				ins->setVersion("1.0");
				ins->setDescription("A STL model library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	void STLInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		factory->addContentAction(std::string("STL"),
			[=](const std::string& path)->std::shared_ptr<Node>
			{
				auto node = std::make_shared<StlLoader<DataType3f>>();
				node->varFileName()->setValue(path);
				return node;
			});

		factory->addContentAction(std::string("stl"),
			[=](const std::string& path)->std::shared_ptr<Node>
			{
				auto node = std::make_shared<StlLoader<DataType3f>>();
				node->varFileName()->setValue(path);
				return node;
			});

		auto page = factory->addPage(
			"IO", 
			"ToolBarIco/Modeling/Modeling.png");

		auto group = page->addGroup("Modeling");

		group->addAction(
			"Stl Loader",
			"ToolBarIco/Modeling/TriangularMesh.png",
			[=]()->std::shared_ptr<Node> { 
				auto node = std::make_shared<StlLoader<DataType3f>>();

				return node; 
			});

	}
}

dyno::PluginEntry* StlLoader::initStaticPlugin()
{
	if (dyno::STLInitializer::instance()->initialize())
		return dyno::STLInitializer::instance();

	return nullptr;
}

PERIDYNO_API dyno::PluginEntry* StlLoader::initDynoPlugin()
{
	if (dyno::STLInitializer::instance()->initialize())
		return dyno::STLInitializer::instance();

	return nullptr;
}

