#pragma once

#include <QMainWindow>

#include "NodeEditor/QtNodeWidget.h"
#include "NodeEditor/QtModuleFlowScene.h"

namespace dyno
{
	class PModuleEditorToolBar;
	class POpenGLWidget;


	class PModuleEditor :
		public QMainWindow
	{
		Q_OBJECT
	public:
		PModuleEditor(Qt::QtNodeWidget* widget, POpenGLWidget* openGLWidget = NULL);

		PModuleEditorToolBar* toolBar() { return mToolBar; }

		Qt::QtModuleFlowScene* moduleFlowScene() { return mModuleFlowScene; }


	signals:
		void changed(Node* node);

	private:
		Qt::QtModuleFlowScene* mModuleFlowScene;
		PModuleEditorToolBar* mToolBar;
		POpenGLWidget* mOpenGLWidget = NULL;

	};
}