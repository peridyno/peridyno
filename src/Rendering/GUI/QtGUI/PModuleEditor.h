#pragma once

#include <QMainWindow>

#include "NodeEditor/QtNodeWidget.h"
#include "NodeEditor/QtModuleFlowScene.h"

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

		Qt::QtModuleFlowScene* moduleFlowScene() { return mModuleFlowScene; }

	signals:
		void changed(Node* node);

	private:
		Qt::QtModuleFlowScene* mModuleFlowScene;
		PModuleEditorToolBar* mToolBar;
	};
}