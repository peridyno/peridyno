#ifndef PMODULELISTWIDGET_H
#define PMODULELISTWIDGET_H

#include <QListWidget>

namespace dyno
{
	class Module;
	class Node;

	class PModuleListItem : public QListWidgetItem
	{
	public:
		PModuleListItem(Module* module, QListWidget *listview = nullptr);

		Module* getModule() { return m_module; }

	private:
		Module* m_module;
	};

	class PModuleListWidget : public QListWidget
	{
		Q_OBJECT

	public:
		PModuleListWidget(QWidget *parent = nullptr);

	Q_SIGNALS:
		void notifyModuleSelected(Module* module);

	public slots:
		void updateModule(Node* node);
		void moduleSelected(QListWidgetItem *item);
	};
}

#endif // PMODULELISTWIDGET_H
