#include "initializeModeling.h"

#include "NodeFactory.h"

#include "CubeModel.h"
#include "SphereModel.h"
#include "SphereSampler.h"
#include "CylinderModel.h"
#include "ConeModel.h"
#include "Turning.h"
#include "Merge.h"

#include "CubeSampler.h"
#include "SphereSampler.h"

#include "StaticTriangularMesh.h"
#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"
#include "Turning.h"

#include "PoissonDiskSampling.h"

namespace dyno 
{
	std::atomic<ModelingInitializer*> ModelingInitializer::gInstance;
	std::mutex ModelingInitializer::gMutex;

	PluginEntry* ModelingInitializer::instance()
	{
		ModelingInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new ModelingInitializer();
				ins->setName("Modeling");
				ins->setVersion("1.0");
				ins->setDescription("A modeling library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	void ModelingInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Modeling", 
			"ToolBarIco/Modeling/Modeling.png");

		auto group = page->addGroup("Modeling");

		group->addAction(
			"Cube",
			"ToolBarIco/Modeling/Cube.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<CubeModel<DataType3f>>();
			});

		group->addAction(
			"Sphere",
			"ToolBarIco/Modeling/Sphere.png",
			[=]()->std::shared_ptr<Node> {
				auto node = std::make_shared<SphereModel<DataType3f>>();

				auto module = std::make_shared<GLSurfaceVisualModule>();
				module->setColor(Vec3f(0.8, 0.52, 0.25));
				module->setVisible(true);
				node->stateTriangleSet()->connect(module->inTriangleSet());
				node->graphicsPipeline()->pushModule(module);

				return node;
			});

		group->addAction(
			"Cylinder",
			"ToolBarIco/Modeling/Cylinder.png",
			[=]()->std::shared_ptr<Node> {
				auto node = std::make_shared<CylinderModel<DataType3f>>();

				auto module = std::make_shared<GLSurfaceVisualModule>();
				module->setColor(Vec3f(0.8, 0.52, 0.25));
				module->setVisible(true);
				node->stateTriangleSet()->connect(module->inTriangleSet());
				node->graphicsPipeline()->pushModule(module);

				return node;
			});
		group->addAction(
			"Cone",
			"ToolBarIco/Modeling/Cone.png",
			[=]()->std::shared_ptr<Node> {
				auto node = std::make_shared<ConeModel<DataType3f>>(); 

				auto module = std::make_shared<GLSurfaceVisualModule>();
				module->setColor(Vec3f(0.8, 0.52, 0.25));
				module->setVisible(true);
				node->stateTriangleSet()->connect(module->inTriangleSet());
				node->graphicsPipeline()->pushModule(module);

				return node;
			});


		group->addAction(
			"Turning Model",
			"ToolBarIco/Modeling/Turn.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<TurningModel<DataType3f>>();
			});


		group->addAction(
			"Merge",
			"ToolBarIco/Modeling/CubeCombo.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<Merge<DataType3f>>();
			});


		group->addAction(
			"Sphere Sampler",
			"ToolBarIco/Modeling/SphereSampler_v3.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<SphereSampler<DataType3f>>();
			});


		group->addAction(
			"Cube Sampler",
			"ToolBarIco/Modeling/CubeSampler.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<CubeSampler<DataType3f>>();
			});

		group->addAction(
			"Poisson Disk Sampler",
			"ToolBarIco/Modeling/PoissonDiskSampler_v2.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<PoissonDiskSampling<DataType3f>>();
			});


	}
}

dyno::PluginEntry* Modeling::initStaticPlugin()
{
	if (dyno::ModelingInitializer::instance()->initialize())
		return dyno::ModelingInitializer::instance();

	return nullptr;
}

PERIDYNO_API dyno::PluginEntry* Modeling::initDynoPlugin()
{
	if (dyno::ModelingInitializer::instance()->initialize())
		return dyno::ModelingInitializer::instance();

	return nullptr;
}
