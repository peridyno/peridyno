#include "WPromptPanel.h"
#include <Wt/WVBoxLayout.h>
#include <Wt/WText.h>
#include <Wt/WTable.h>

WPromptPanel::WPromptPanel()
{
}

WPromptPanel::~WPromptPanel()
{
}

void WPromptPanel::createPromptPanel(Wt::WContainerWidget* promptNodeWidget, std::map<std::string, std::tuple<std::string, int>> promptNodes)
{
	auto layout = promptNodeWidget->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);

	auto promptPanel = layout->addWidget(std::make_unique<Wt::WPanel>());
	promptPanel->setTitle("Prompt Node");
	promptPanel->setCollapsible(true);
	promptPanel->setStyleClass("scrollable-content");
	promptPanel->setMargin(0);
	mPromptTree = promptPanel->setCentralWidget(std::make_unique<Wt::WTreeView>());

	//mPromptTree->setModel();

	//mPromptTable->setMargin(0);
	//mPromptTable->setSortingEnabled(false);
	//mPromptTable->setSelectionMode(Wt::SelectionMode::Single);
	//mPromptTable->setEditTriggers(Wt::EditTrigger::None);
	//mPromptTable->setColumnResizeEnabled(true);
	//mPromptTable->setColumnWidth(0, 100);
	//mPromptTable->setColumnWidth(1, 280);
	//mPromptTable->setSortingEnabled(false);

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
