#pragma once

#include <Wt/WContainerWidget.h>
#include <Wt/WPanel.h>
#include <Wt/WTreeView.h>

#include <Wt/WAbstractItemModel.h>
#include <Wt/WAbstractTableModel.h>

#include <Wt/WText.h>
#include <Wt/WTree.h>
#include <Wt/WTreeTable.h>
#include <Wt/WTreeTableNode.h>

class WPromptPanel
{
public:
	WPromptPanel();
	~WPromptPanel();

	void createPromptPanel(Wt::WContainerWidget* promptNodeWidget, std::map<std::string, std::tuple<std::string, int>> promptNodes);

	Wt::Signal<std::tuple<std::string, int>>& addPromptNode() { return _addPromptNode; };

private:
	Wt::WPanel* mPromptPanel;

	Wt::WTreeView* mPromptTree;

	Wt::Signal<std::tuple<std::string, int>> _addPromptNode;
};