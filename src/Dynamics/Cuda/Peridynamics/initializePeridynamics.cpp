#include "initializePeridynamics.h"

#include "Cloth.h"

#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"

#include "NodeFactory.h"

namespace dyno
{
	std::atomic<PeridynamicsInitializer*> PeridynamicsInitializer::gInstance;
	std::mutex PeridynamicsInitializer::gMutex;

	PluginEntry* PeridynamicsInitializer::instance()
	{
		PeridynamicsInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new PeridynamicsInitializer();
				ins->setName("Peridynamics");
				ins->setVersion("1.0");
				ins->setDescription("A peridynamics library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	PeridynamicsInitializer::PeridynamicsInitializer()
		: PluginEntry()
	{
	}

	void PeridynamicsInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Soft Body",
			"ToolBarIco/SoftBody/SoftBody.png");

		auto group = page->addGroup("Soft Body");

		group->addAction(
			"Cloth",
			"ToolBarIco/SoftBody/SoftBody2.png",
			[=]()->std::shared_ptr<Node> {

				auto cloth = std::make_shared<Cloth<DataType3f>>();
				cloth->setDt(0.001f);

				auto pointRenderer = std::make_shared<GLPointVisualModule>();
				pointRenderer->setColor(Color(1, 0.2, 1));
				pointRenderer->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);
				pointRenderer->varPointSize()->setValue(0.002f);
				cloth->stateTriangleSet()->connect(pointRenderer->inPointSet());
				cloth->stateVelocity()->connect(pointRenderer->inColor());

				cloth->graphicsPipeline()->pushModule(pointRenderer);
				cloth->setVisible(true);

				auto wireRenderer = std::make_shared<GLWireframeVisualModule>();
				wireRenderer->varBaseColor()->setValue(Color(1.0, 0.8, 0.8));
				wireRenderer->varRadius()->setValue(0.001f);
				wireRenderer->varRenderMode()->setCurrentKey(GLWireframeVisualModule::CYLINDER);
				cloth->stateTriangleSet()->connect(wireRenderer->inEdgeSet());
				cloth->graphicsPipeline()->pushModule(wireRenderer);

				auto surfaceRenderer = std::make_shared<GLSurfaceVisualModule>();
				cloth->stateTriangleSet()->connect(surfaceRenderer->inTriangleSet());
				cloth->graphicsPipeline()->pushModule(surfaceRenderer);

				return cloth;
			});

		group->addAction(
			"Soft Body",
			"ToolBarIco/SoftBody/SoftBody5.png",
			[=]()->std::shared_ptr<Node> {
				return nullptr;
			});
	}
}

PERIDYNO_API dyno::PluginEntry* Peridynamics::initDynoPlugin()
{
	if (dyno::PeridynamicsInitializer::instance()->initialize())
		return dyno::PeridynamicsInitializer::instance();

	return nullptr;
}

dyno::PluginEntry* Peridynamics::initStaticPlugin()
{
	if (dyno::PeridynamicsInitializer::instance()->initialize())
		return dyno::PeridynamicsInitializer::instance();

	return nullptr;
}
