#include "initializeModeling.h"

#include "NodeFactory.h"

#include "CubeModel.h"
#include "SphereModel.h"

#include "CubeSampler.h"

#include "StaticTriangularMesh.h"
#include "GLSurfaceVisualModule.h"

namespace dyno 
{
	ModelingInitializer::ModelingInitializer()
	{
		initializeNodeCreators();
	}

	void ModelingInitializer::initializeNodeCreators()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto group = factory->addGroup(
			"Modeling", 
			"Modeling", 
			"ToolBarIco/FiniteElement/FiniteElement.png");

		group->addAction(
			"Cube",
			"48px-Image-x-generic.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<CubeModel<DataType3f>>();
			});

		group->addAction(
			"Sphere",
			"48px-Image-x-generic.png",
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
			"Cube Sampler",
			"48px-Image-x-generic.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<CubeSampler<DataType3f>>();
			});

		group->addAction(
			"Triangular Mesh",
			"48px-Image-x-generic.png",
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