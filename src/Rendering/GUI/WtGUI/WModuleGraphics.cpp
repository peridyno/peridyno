#include "WModuleGraphics.h"


WModuleGraphics::WModuleGraphics()
{
	layout = this->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	this->setMargin(0);

	// add node
	addPanel = layout->addWidget(std::make_unique<Wt::WPanel>());
	addPanel->setTitle("Add Node");
	addPanel->setCollapsible(false);

	// node graphics
	modulePanel = layout->addWidget(std::make_unique<Wt::WPanel>());
	modulePanel->setTitleBar(false);
	modulePanel->setCollapsible(false);
	modulePanel->setMargin(0);
}

WModuleGraphics::~WModuleGraphics() {}