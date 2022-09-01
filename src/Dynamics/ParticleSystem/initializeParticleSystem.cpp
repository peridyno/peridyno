#include "initializeParticleSystem.h"

#include "Module/LinearDamping.h"
#include "Module/ParticleIntegrator.h"
#include "Module/ImplicitViscosity.h"
#include "Module/SummationDensity.h"
#include "Module/DensityPBD.h"
#include "Module/BoundaryConstraint.h"
#include "Module/VariationalApproximateProjection.h"

#include "ParticleSystem/CircularEmitter.h"
#include "ParticleSystem/SquareEmitter.h"

#include "GLWireframeVisualModule.h"

#include "ParticleFluid.h"
#include "StaticBoundary.h"

#include "NodeFactory.h"

namespace dyno 
{
	ParticleSystemInitializer::ParticleSystemInitializer()
	{
		TypeInfo::New<LinearDamping<DataType3f>>();
		TypeInfo::New<ParticleIntegrator<DataType3f>>();
		TypeInfo::New<ImplicitViscosity<DataType3f>>();
		TypeInfo::New<DensityPBD<DataType3f>>();
		TypeInfo::New<SummationDensity<DataType3f>>();
		TypeInfo::New<VariationalApproximateProjection<DataType3f>>();
		//TypeInfo::New<BoundaryConstraint<DataType3f>>();

		initializeNodeCreators();
	}

	void ParticleSystemInitializer::initializeNodeCreators()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Particle System", 
			"ToolBarIco/ParticleSystem/ParticleSystem.png");

		auto group = page->addGroup("Particle System");

		group->addAction(
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

		group->addAction(
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

		group->addAction(
			"Particle Fluid",
			"ToolBarIco/ParticleSystem/ParticleFluid.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<ParticleFluid<DataType3f>>(); });

		group->addAction(
			"Boundary",
			"ToolBarIco/RigidBody/StaticBoundary.png",
			[=]()->std::shared_ptr<Node> { 
				auto  boundary = std::make_shared<StaticBoundary<DataType3f>>();
				boundary->loadCube(Vec3f(-0.5, 0, -0.5), Vec3f(0.5, 1, 0.5), 0.02, true);
				return boundary; });
	}

}