#ifndef QCONSOLEWIDGET_H
#define QCONSOLEWIDGET_H

#include <QWidget>
#include <QFileSystemModel>
#include <QTreeView>
#include <QListView>


namespace dyno
{
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

	public slots:
		void treeItemSelected(const QModelIndex& index);

	private:
		QFileSystemModel* model;
		QFileSystemModel* listModel;
		QTreeView* treeView;
		QListView* listView;
	};
}

#endif // QCONSOLEWIDGET_H
