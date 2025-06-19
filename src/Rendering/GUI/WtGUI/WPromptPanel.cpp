#include "WPromptPanel.h"
#include <Wt/WVBoxLayout.h>
#include <Wt/WSortFilterProxyModel.h>

#include <WSceneDataModel.h>
#include <Wt/WAny.h>
#include <any>



WPromptPanel::WPromptPanel()
{
}

WPromptPanel::~WPromptPanel()
{
}

void WPromptPanel::createPromptPanel(Wt::WContainerWidget* promptNodeWidget, std::map<std::string, connectionData> promptNodes)
{
	std::shared_ptr<WPromptNode> nodeModel = std::make_shared<WPromptNode>(promptNodes);

	canConnectionDatas = promptNodes;

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

	mPromptTree->doubleClicked().connect([=](Wt::WModelIndex index, Wt::WMouseEvent event)
		{
			try {
				auto name = Wt::asString(proxy->data(index.row(), 1)).toUTF8();
				if (canConnectionDatas.find(name) != canConnectionDatas.end())
				{
					_addPromptNode.emit(canConnectionDatas.find(name)->second);
					clear();
				}
			}
			catch (const std::bad_any_cast& e) {
				std::cerr << "×ª»»Ê§°Ü: " << e.what() << '\n';
			}
		});
}

void WPromptPanel::clear()
{
	std::shared_ptr<WPromptNode> emptyModel = std::make_shared<WPromptNode>();
	mPromptTree->setModel(emptyModel);
}
