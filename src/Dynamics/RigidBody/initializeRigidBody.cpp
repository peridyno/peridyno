#include "initializeRigidBody.h"

#include "NodeFactory.h"

namespace dyno
{
	RigidBodyInitializer::RigidBodyInitializer()
	{
		initializeNodeCreators();
	}

	void RigidBodyInitializer::initializeNodeCreators()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Rigid Body Dynamics",
			"ToolBarIco/ParticleSystem/ParticleSystem.png");

		auto group = page->addGroup("Rigid Body Dynamics");

		group->addAction(
			"GhostFluid",
			"ToolBarIco/RigidBody/GhostFluid.png",
			nullptr);

		group->addAction(
			"GhostParticles",
			"ToolBarIco/RigidBody/GhostParticles.png",
			nullptr);

		group->addAction(
			"StaticBoundary",
			"ToolBarIco/RigidBody/StaticBoundary.png",
			nullptr);
	}

}