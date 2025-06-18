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

	mPromptTree->doubleClicked().connect([=](Wt::WModelIndex index, Wt::WMouseEvent event)
		{
			try {
				auto type = Wt::asString(proxy->data(index.row(), 0)).toUTF8();
				auto name = Wt::asString(proxy->data(index.row(), 1)).toUTF8();
				auto connectIndex = Wt::asNumber(proxy->data(index.row(), 2));

				std::tuple<std::string, int> addNode;
				std::get<0>(addNode) = name;
				std::get<1>(addNode) = connectIndex;

				_addPromptNode.emit(addNode);
			}
			catch (const std::bad_any_cast& e) {
				std::cerr << "×ª»»Ê§°Ü: " << e.what() << '\n';
			}
		});
}