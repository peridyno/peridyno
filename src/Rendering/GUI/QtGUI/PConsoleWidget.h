#ifndef QCONSOLEWIDGET_H
#define QCONSOLEWIDGET_H

#include <QWidget>
#include <QFileSystemModel>
#include <QTreeView>
#include <QListView>


namespace dyno
{
	class Node;

	class PConsoleWidget : public QWidget
	{
		Q_OBJECT
	public:
		explicit PConsoleWidget(QWidget *parent = nullptr);

	signals:

	public slots:
	};

	class QContentBrowser : public QWidget
	{
		Q_OBJECT
	public:
		explicit QContentBrowser(QWidget* parent = nullptr);

	signals:

	Q_SIGNALS:
		void nodeCreated(std::shared_ptr<Node> node);

	public slots:
		void treeItemSelected(const QModelIndex& index);
		void assetItemSelected(const QModelIndex& index);
		void assetDoubleClicked(const QModelIndex& index);

	private:
		QFileSystemModel* model;
		QFileSystemModel* listModel;
		QTreeView* treeView;
		QListView* listView;
	};
}

#endif // QCONSOLEWIDGET_H
