#pragma once

#include <QMainWindow>

#include "NodeEditor/QtNodeWidget.h"

namespace dyno
{
	class PModuleEditorToolBar;

	class PModuleEditor :
		public QMainWindow
	{
		Q_OBJECT
	public:
		PModuleEditor(Qt::QtNodeWidget* widget);

		PModuleEditorToolBar* toolBar() { return mToolBar; }

	signals:
		void changed(Node* node);

	private:
		PModuleEditorToolBar* mToolBar;
	};
}