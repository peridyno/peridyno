#include "WNodeGraphics.h"


WNodeGraphics::WNodeGraphics()
{
	layout = this->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	this->setMargin(0);

	// add node
	addPanel = layout->addWidget(std::make_unique<Wt::WPanel>());
	addPanel->setTitle("Add Node");
	addPanel->setCollapsible(false);

	// node graphics
	nodePanel = layout->addWidget(std::make_unique<Wt::WPanel>());
	nodePanel->setTitleBar(false);
	nodePanel->setCollapsible(false);
	nodePanel->setMargin(0);
}

WNodeGraphics::~WNodeGraphics() {}