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
#include <NodeEditor/WtFlowWidget.h>

class WPromptPanel
{
public:
	WPromptPanel();
	~WPromptPanel();

	void createPromptPanel(Wt::WContainerWidget* promptNodeWidget, std::map<std::string, connectionData> promptNodes);
	
	void clear();

	Wt::Signal<connectionData>& addPromptNode() { return _addPromptNode; };

private:
	Wt::WPanel* mPromptPanel;

	Wt::WTreeView* mPromptTree;

	Wt::Signal<connectionData> _addPromptNode;

	std::map<std::string, connectionData> canConnectionDatas;

};