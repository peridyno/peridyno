#include "initializeModeling.h"

#include "NodeFactory.h"

#include "BasicShapes/PlaneModel.h"
#include "BasicShapes/CubeModel.h"
#include "BasicShapes/SphereModel.h"
#include "BasicShapes/CylinderModel.h"
#include "BasicShapes/ConeModel.h"
#include "BasicShapes/CapsuleModel.h"

#include "Commands/Turning.h"
#include "Commands/Merge.h"
#include "Commands/Turning.h"

#include "Samplers/ShapeSampler.h"

#include "StaticMeshLoader.h"

#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"

#include "Mapping/MergeTriangleSet.h"

#include "GltfLoader.h"

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

		factory->addContentAction(std::string("gltf"),
			[=](const std::string& path)->std::shared_ptr<Node>
			{
				auto node = std::make_shared<GltfLoader<DataType3f>>();
				node->varFileName()->setValue(path);
				return node;
			});

		factory->addContentAction(std::string("glb"),
			[=](const std::string& path)->std::shared_ptr<Node>
			{
				auto node = std::make_shared<GltfLoader<DataType3f>>();
				node->varFileName()->setValue(path);
				return node;
			});


		auto page = factory->addPage(
			"Modeling", 
			"ToolBarIco/Modeling/Modeling.png");

		auto basic = page->addGroup("Basic");

		basic->addAction(
			"Plane",
			"ToolBarIco/Modeling/Plane.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<PlaneModel<DataType3f>>();
			});

		basic->addAction(
			"Cube",
			"ToolBarIco/Modeling/Cube.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<CubeModel<DataType3f>>();
			});

		basic->addAction(
			"Sphere",
			"ToolBarIco/Modeling/Sphere.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<SphereModel<DataType3f>>();
			});

		basic->addAction(
			"Cylinder",
			"ToolBarIco/Modeling/Cylinder.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<CylinderModel<DataType3f>>();
			});
		basic->addAction(
			"Cone",
			"ToolBarIco/Modeling/Cone.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<ConeModel<DataType3f>>();
			});

		basic->addAction(
			"Capsule",
			"ToolBarIco/Modeling/Capsule.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<CapsuleModel<DataType3f>>();
			});

		auto op = page->addGroup("Operation");

		op->addAction(
			"Turning Model",
			"ToolBarIco/Modeling/Turn.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<TurningModel<DataType3f>>();
			});


		op->addAction(
			"Merge",
			"ToolBarIco/Modeling/CubeCombo.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<MergeTriangleSet<DataType3f>>();
			});

		auto samplers = page->addGroup("Samplers");

// 		samplers->addAction(
// 			"Sphere Sampler",
// 			"ToolBarIco/Modeling/SphereSampler_v3.png",
// 			[=]()->std::shared_ptr<Node> {
// 				return std::make_shared<Shape<DataType3f>>();
// 			});

		samplers->addAction(
			"Sampler",
			"ToolBarIco/Modeling/CubeSampler.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<ShapeSampler<DataType3f>>();
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
