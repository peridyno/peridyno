#include "initializeModeling.h"

#include "NodeFactory.h"

#include "CubeModel.h"
#include "SphereModel.h"

#include "CubeSampler.h"

#include "StaticTriangularMesh.h"
#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"

#include "PoissonDiskSampling.h"

#include "ParticleSystem/CircularEmitter.h"
#include "ParticleSystem/SquareEmitter.h"

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
			"ToolBarIco/FiniteElement/FiniteElement.png");

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
			"Cube Sampler",
			"ToolBarIco/Modeling/CubeSampler.png",
			[=]()->std::shared_ptr<Node> {
				return std::make_shared<CubeSampler<DataType3f>>();
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

		auto particlePage = factory->addPage(
			"Particle System",
			"ToolBarIco/ParticleSystem/ParticleSystem.png");

		auto particleGroup = particlePage->addGroup("Particle System");

		particleGroup->addAction(
			"Circular Emitter",
			"ToolBarIco/ParticleSystem/ParticleEmitterRound.png",
			[=]()->std::shared_ptr<Node> {
				auto emitter = std::make_shared<CircularEmitter<DataType3f>>();

				auto wireRender = std::make_shared<GLWireframeVisualModule>();
				wireRender->setColor(Vec3f(0, 1, 0));
				emitter->stateOutline()->connect(wireRender->inEdgeSet());
				emitter->graphicsPipeline()->pushModule(wireRender);
				return emitter;
			});

		particleGroup->addAction(
			"Square Emitter",
			"ToolBarIco/ParticleSystem/ParticleEmitterSquare.png",
			[=]()->std::shared_ptr<Node> { 
				auto emitter = std::make_shared<SquareEmitter<DataType3f>>();

				auto wireRender = std::make_shared<GLWireframeVisualModule>();
				wireRender->setColor(Vec3f(0, 1, 0));
				emitter->stateOutline()->connect(wireRender->inEdgeSet());
				emitter->graphicsPipeline()->pushModule(wireRender);
				return emitter;;
			});
	}
}