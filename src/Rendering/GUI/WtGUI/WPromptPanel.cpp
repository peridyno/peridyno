#include "WPromptPanel.h"
#include <Wt/WVBoxLayout.h>
#include <Wt/WText.h>
#include <Wt/WTable.h>
#include <WSceneDataModel.h>

WPromptPanel::WPromptPanel()
{
}

WPromptPanel::~WPromptPanel()
{
}

void WPromptPanel::createPromptPanel(Wt::WContainerWidget* promptNodeWidget, std::map<std::string, std::tuple<std::string, int>> promptNodes)
{
	std::shared_ptr<WPromptNode> nodeModel = std::make_shared<WPromptNode>(promptNodes);

	auto layout = promptNodeWidget->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);

	mPromptPanel = layout->addWidget(std::make_unique<Wt::WPanel>());
	mPromptPanel->setTitle("Prompt Node");
	mPromptPanel->setCollapsible(true);
	mPromptPanel->setStyleClass("scrollable-content");
	mPromptPanel->setMargin(0);
	//mPromptPanel->collapse();

	mPromptTree = mPromptPanel->setCentralWidget(std::make_unique<Wt::WTreeView>());
	mPromptTree->setModel(nodeModel);
	mPromptTree->setMargin(0);
	mPromptTree->setSortingEnabled(true);
	mPromptTree->setSelectionMode(Wt::SelectionMode::Single);
	mPromptTree->setEditTriggers(Wt::EditTrigger::None);
	mPromptTree->setColumnResizeEnabled(true);
	mPromptTree->setColumnWidth(0, 250);
	mPromptTree->setColumnWidth(1, 250);

	//auto i = 0;
	//for (auto promptNode : promptNodes)
	//{
	//	mPromptTable->elementAt(i, 0)->addWidget(std::make_unique<Wt::WText>(promptNode.first));
	//	i++;
	//}
}

void WPromptPanel::emitAddNode()
{
}