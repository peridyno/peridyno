#include "PSceneGraphWidget.h"

#include "Framework/SceneGraph.h"
#include "PInsertNodeAction.h"
#include "Framework/Node.h"

#include <iostream>

#include <QAction>
#include <QMenu>
#include <QObject>

namespace dyno
{
	PSceneGraphWidgetItem::PSceneGraphWidgetItem(Node* node, QTreeWidget *treeview) :
		QTreeWidgetItem(treeview),
		m_node(node)
	{
		if (node != nullptr)
		{
			setText(0, m_node->getName().c_str());
		}
	}

	PSceneGraphWidgetItem::PSceneGraphWidgetItem(Node* node, QTreeWidgetItem *parent) :
		QTreeWidgetItem(parent),
		m_node(node)
	{
		if (node != nullptr)
		{
			setText(0, m_node->getName().c_str());
		}
	}


	PSceneGraphWidget::PSceneGraphWidget(QWidget *parent) :
		QTreeWidget(parent)
	{
		this->setMinimumWidth(100);
		this->setHeaderLabel("Tree Nodes:");

		this->setContextMenuPolicy(Qt::CustomContextMenu);

		this->setExpandsOnDoubleClick(false);


		connect(this, SIGNAL(itemClicked(QTreeWidgetItem*, int)), this, SLOT(nodeSelected(QTreeWidgetItem*, int)));
		connect(this, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(nodeDoubleClicked(QTreeWidgetItem*, int)));
		connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(popMenu(const QPoint&)));

		emit updateTree();
	}

	void PSceneGraphWidget::updateTree()
	{
		clear();

		SceneGraph& scenegraph = SceneGraph::getInstance();
		std::shared_ptr<Node> root = scenegraph.getRootNode();

		if(root != nullptr)
			root->traverseTopDown<PInsertTreeNodeAction>(this);

//		PSceneGraphNode* root = new PSceneGraphNode(scenegraph.getRootNode(), this);
	}

	void PSceneGraphWidget::nodeClicked(QTreeWidgetItem* item, int index)
	{
	}

	void PSceneGraphWidget::nodeDoubleClicked(QTreeWidgetItem* item, int index)
	{
		PSceneGraphWidgetItem* nodeItem = dynamic_cast<PSceneGraphWidgetItem*>(item);
		if (nodeItem != nullptr)
		{
			emit notifyNodeDoubleClicked(nodeItem->getNode());
		}
	}

	void PSceneGraphWidget::popMenu(const QPoint& pos)
	{
		QTreeWidgetItem* curItem = this->currentItem();
		if (curItem == nullptr)
			return;

		QAction deleteItem(QString::fromLocal8Bit("Delete"), this);
		//connect(&deleteItem, SIGNAL(triggered()), this, SLOT(deleteItem()));
		QMenu menu(this);
		menu.addAction(&deleteItem);
		menu.exec(QCursor::pos());  //在当前鼠标位置显示
	}

	void PSceneGraphWidget::nodeSelected(QTreeWidgetItem *item, int column)
	{
		PSceneGraphWidgetItem* nodeItem = dynamic_cast<PSceneGraphWidgetItem*>(item);
		if (nodeItem != nullptr)
		{
			emit notifyNodeSelected(nodeItem->getNode());
		}
	}

}