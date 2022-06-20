#include "initializeInteraction.h"

#include "NodeFactory.h"

namespace dyno 
{
	InteractionInitializer::InteractionInitializer()
	{
		initializeNodeCreators();
	}

	void InteractionInitializer::initializeNodeCreators()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto group = factory->addGroup(
			"Interaction", 
			"Interaction", 
			"ToolBarIco/FiniteElement/FiniteElement.png");
	}
}