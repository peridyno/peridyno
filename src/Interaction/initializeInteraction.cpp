#include "initializeInteraction.h"

#include "NodeFactory.h"

#include "PickerNode.h"

namespace dyno 
{
	InteractionInitializer::InteractionInitializer()
	{
		initializeNodeCreators();
	}

	void InteractionInitializer::initializeNodeCreators()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Interaction", 
			"ToolBarIco/Interaction/Interaction.png");

		auto group = page->addGroup("Interaction");

		group->addAction(
			"Picker",
			"ToolBarIco/Interaction/Picker.png",//48px-Image-x-generic.png
			[=]()->std::shared_ptr<Node> { return std::make_shared<PickerNode<DataType3f>>(); });
	}
}