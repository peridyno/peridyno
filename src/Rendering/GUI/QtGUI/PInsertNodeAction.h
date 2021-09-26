#ifndef PINSERTTREENODEACTION_H
#define PINSERTTREENODEACTION_H

#include <stack>
#include "Action.h"

#include <QTreeWidget>

namespace dyno
{
	class Node;
	class PSceneGraphWidgetItem;

	class PInsertTreeNodeAction : public Action
	{
	public:
		PInsertTreeNodeAction(QTreeWidget* widget);
		virtual ~PInsertTreeNodeAction() {};

	public:
		void start(Node* node) override;
		void end(Node* node) override;

	private:
		QTreeWidget* m_treeWidget;

		std::stack<PSceneGraphWidgetItem*> treeItemStack;
	};
}

#endif // QTREEWIDGETNODEITEM_H
