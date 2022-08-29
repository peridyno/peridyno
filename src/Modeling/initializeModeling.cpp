#include "initializeModeling.h"

#include "NodeFactory.h"

#include "CubeModel.h"
#include "SphereModel.h"
#include "SphereSampler.h"

#include "CubeSampler.h"

#include "StaticTriangularMesh.h"
#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"

#include "PoissonDiskSampling.h"

namespace dyno 
{
	ModelingInitializer::ModelingInitializer()
	{
		initializeNodeCreators();
	}

	void ModelingInitializer::initializeNodeCreators()
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
				return std::make_shared<PoissonDiksSampling<DataType3f>>();
			});

		group->addAction(
			"Triangular Mesh",
			"ToolBarIco/Modeling/TriangularMesh.png",
			[=]()->std::shared_ptr<Node> { 
				auto node = std::make_shared<StaticTriangularMesh<DataType3f>>();

				auto module = std::make_shared<GLSurfaceVisualModule>();
				module->setColor(Vec3f(0.8, 0.52, 0.25));
				module->setVisible(true);
				node->stateTopology()->connect(module->inTriangleSet());
				node->graphicsPipeline()->pushModule(module);

				return node; 
			});
	}
}