#include "PInsertNodeAction.h"

#include "Framework/Node.h"
#include "PSceneGraphWidget.h"

namespace dyno
{
	PInsertTreeNodeAction::PInsertTreeNodeAction(QTreeWidget* widget) :
		m_treeWidget(widget)
	{

	}

	void PInsertTreeNodeAction::start(Node* node)
	{
		PSceneGraphWidgetItem* parentItem = nullptr;
		if (!PInsertTreeNodeAction::treeItemStack.empty())
		{
			parentItem = treeItemStack.top();
		}

		PSceneGraphWidgetItem* childItem;
		if (parentItem == nullptr)
		{
			childItem = new PSceneGraphWidgetItem(node, m_treeWidget);
		}
		else
		{
			childItem = new PSceneGraphWidgetItem(node, parentItem);
		}

		PInsertTreeNodeAction::treeItemStack.push(childItem);
	}

	void PInsertTreeNodeAction::end(Node* node)
	{
		treeItemStack.pop();
	}

}
