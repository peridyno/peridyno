#include "WPromptPanel.h"
#include <Wt/WVBoxLayout.h>
#include <Wt/WSortFilterProxyModel.h>

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

	// sort model
	auto proxy = std::make_shared<Wt::WSortFilterProxyModel>();
	proxy->setSourceModel(nodeModel);
	proxy->setDynamicSortFilter(true);
	proxy->setSortRole(Wt::ItemDataRole::Display);
	proxy->sort(0, Wt::SortOrder::Ascending);

	mPromptTree->setModel(proxy);
	mPromptTree->setMargin(0);
	mPromptTree->setSortingEnabled(false);
	mPromptTree->setSelectionMode(Wt::SelectionMode::Single);
	mPromptTree->setEditTriggers(Wt::EditTrigger::None);
	mPromptTree->setColumnResizeEnabled(true);
	mPromptTree->setColumnWidth(0, 250);
	mPromptTree->setColumnWidth(1, 250);

	mPromptTree->doubleClicked().connect([=]
		{
			std::cout << "doubleClicked" << std::endl;
		});
}

void WPromptPanel::emitAddNode()
{
}

