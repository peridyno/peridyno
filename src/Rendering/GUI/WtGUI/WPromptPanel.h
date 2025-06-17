#pragma once

#include <Wt/WContainerWidget.h>
#include <Wt/WPanel.h>
#include <Wt/WTreeView.h>
#include <Wt/WAbstractItemModel.h>
#include <Wt/WAbstractTableModel.h>

class WPromptPanel
{
public:
	WPromptPanel();
	~WPromptPanel();

	void createPromptPanel(Wt::WContainerWidget* promptNodeWidget, std::map<std::string, std::tuple<std::string, int>> promptNodes);

	void emitAddNode();

	Wt::Signal<std::pair<std::string, int>>& addPromptNode() { return _addPromptNode; };

private:
	Wt::WPanel* mPromptPanel;

	Wt::WTreeView* mPromptTree;

	Wt::Signal<std::pair<std::string, int>> _addPromptNode;
};