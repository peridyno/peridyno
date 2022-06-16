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

		auto group = factory->addGroup(
			"Rigid Body Dynamics",
			"Rigid Body Dynamics",
			"ToolBarIco/ParticleSystem/ParticleSystem.png");

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