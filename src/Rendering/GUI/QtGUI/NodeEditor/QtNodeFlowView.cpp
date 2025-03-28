#include "QtNodeFlowView.h"

#include <QtWidgets>

#include "nodes/QFlowScene"

namespace Qt
{
	QtNodeFlowView::QtNodeFlowView(QWidget *parent)
		: QtFlowView(parent)
	{
	}

	QtNodeFlowView::QtNodeFlowView(QtFlowScene *scene, QWidget *parent)
		: QtFlowView(scene, parent)
	{
		connect(scene, &QtFlowScene::outPortConextMenu, this, &QtNodeFlowView::showPortContextMenu);
	}

	void QtNodeFlowView::showPortContextMenu(QtNode& n, const PortIndex index, const QPointF& pos)
	{
		QMenu modelMenu;

		auto scenePos = this->mapToScene(this->mapFromGlobal(QPoint(pos.x(), pos.y())));

		//Filter classes
		std::set<QString> categoryFiltered;
		std::unordered_map<QString, QString> registeredModelsCategoryFiltered;

		auto outData = n.nodeDataModel()->outData(index);
		for (auto const& assoc : scene()->registry().registeredModelsCategoryAssociation())
		{
			auto modelName = QString(assoc.first);

			auto type = scene()->registry().create(modelName);

			if (type)
			{
				auto num = type->nPorts(PortType::In);

				for (uint i = 0; i < num; i++)
				{
					if (type->tryInData(i, outData))
					{
						categoryFiltered.insert(assoc.second);
						registeredModelsCategoryFiltered[modelName] = assoc.second;

						continue;
					}
				}
			}
		}

		//Add filterbox to the context menu
		auto* txtBox = new QLineEdit(&modelMenu);
		txtBox->grabKeyboard();

		txtBox->setPlaceholderText(QStringLiteral("Filter"));
		txtBox->setClearButtonEnabled(true);

		auto* txtBoxAction = new QWidgetAction(&modelMenu);
		txtBoxAction->setDefaultWidget(txtBox);

		modelMenu.addAction(txtBoxAction);
		modelMenu.addSeparator();

		auto skipText = QStringLiteral("skip me");

		//Show context menu
		{
			//Add result treeview to the context menu
			auto* treeView = new QTreeWidget(&modelMenu);
			treeView->header()->close();

			auto* treeViewAction = new QWidgetAction(&modelMenu);
			treeViewAction->setDefaultWidget(treeView);

			modelMenu.addAction(treeViewAction);

			QMap<QString, QTreeWidgetItem*> topLevelItems;
			for (auto const& cat : categoryFiltered)
			{
				auto item = new QTreeWidgetItem(treeView);
				item->setText(0, cat);
				item->setData(0, Qt::UserRole, skipText);
				topLevelItems[cat] = item;
			}

			for (auto const& assoc : registeredModelsCategoryFiltered)
			{
				auto parent = topLevelItems[assoc.second];
				auto item = new QTreeWidgetItem(parent);
				item->setText(0, assoc.first);
				item->setData(0, Qt::UserRole, assoc.first);
			}

			treeView->expandAll();

			connect(treeView, &QTreeWidget::itemClicked, [&](QTreeWidgetItem* item, int)
				{
					QString modelName = item->data(0, Qt::UserRole).toString();

					if (modelName == skipText)
					{
						return;
					}

					auto type = scene()->registry().create(modelName);

					if (type)
					{
						auto& node = scene()->createNode(std::move(type));

						node.nodeGraphicsObject().setPos(scenePos);

						scene()->nodePlaced(node);

						//Create connection
						{
							auto num = node.nodeDataModel()->nPorts(PortType::In);

							for (uint i = 0; i < num; i++)
							{
								//TODO: if multiple ports exist, how to choose the best one
								if (node.nodeDataModel()->tryInData(i, outData))
								{
									scene()->createConnection(node, i, n, index);
									break;
								}
							}
						}
					}
					else
					{
						qDebug() << "Model not found";
					}

					modelMenu.close();
				});

			//Setup filtering
			connect(txtBox, &QLineEdit::textChanged, [&](const QString& text)
				{
					for (auto& topLvlItem : topLevelItems)
					{
						bool topItemMatch = false;
						for (int i = 0; i < topLvlItem->childCount(); ++i)
						{
							auto child = topLvlItem->child(i);
							auto modelName = child->data(0, Qt::UserRole).toString();
							const bool match = (modelName.contains(text, Qt::CaseInsensitive));
							child->setHidden(!match);

							topItemMatch |= match;
						}
						//If no child is matched, hide the top level item
						topLvlItem->setHidden(!topItemMatch);
					}
				});

			// make sure the text box gets focus so the user doesn't have to click on it
			txtBox->setFocus();

			categoryFiltered.clear();
			registeredModelsCategoryFiltered.clear();

			modelMenu.exec(QPoint(pos.x(), pos.y()));
		}
	}
}
