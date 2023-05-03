#include "initializeHeightField.h"

#include "Ocean.h"
#include "CapillaryWave.h"
#include "OceanPatch.h"

#include "Mapping/HeightFieldToTriangleSet.h"
#include "GLSurfaceVisualModule.h"

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
			[=]()->std::shared_ptr<Node> { 
				auto patch = std::make_shared<OceanPatch<DataType3f>>();

				auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
				patch->stateHeightField()->connect(mapper->inHeightField());
				patch->graphicsPipeline()->pushModule(mapper);

				auto sRender = std::make_shared<GLSurfaceVisualModule>();
				sRender->setColor(Color(0, 0.2, 1.0));
				sRender->varUseVertexNormal()->setValue(true);
				mapper->outTriangleSet()->connect(sRender->inTriangleSet());
				patch->graphicsPipeline()->pushModule(sRender);

				return patch;
			});

		group->addAction(
			"Ocean",
			"ToolBarIco/HeightField/Ocean.png",
			[=]()->std::shared_ptr<Node> { 
				auto ocean = std::make_shared<Ocean<DataType3f>>();
				ocean->varExtentX()->setValue(2);
				ocean->varExtentZ()->setValue(2);

				auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
				ocean->stateHeightField()->connect(mapper->inHeightField());
				ocean->graphicsPipeline()->pushModule(mapper);

				auto sRender = std::make_shared<GLSurfaceVisualModule>();
				sRender->setColor(Color(0, 0.2, 1.0));
				mapper->outTriangleSet()->connect(sRender->inTriangleSet());
				ocean->graphicsPipeline()->pushModule(sRender);

				return ocean;
			});

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
