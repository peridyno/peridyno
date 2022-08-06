#pragma once

#include "ToolBar/TabToolbar.h"
#include "NodeEditor/QtNodeFlowWidget.h"

namespace dyno
{
	class Node;

	class PMainToolBar : public tt::TabToolbar
	{
		Q_OBJECT
	public:
		PMainToolBar(Qt::QtNodeFlowWidget* nodeFlow, QWidget* parent = nullptr, unsigned _groupMaxHeight = 75, unsigned _groupRowCount = 3);


	Q_SIGNALS:
		void newSceneLoaded();
		void nodeCreated(std::shared_ptr<Node> node);

	public slots:
		void newFile();
		void openFile();
		void saveFile();
		void saveAsFile();
		void closeFile();
		void closeAllFiles();

	private:

		void setupFileMenu();

		Qt::QtNodeFlowWidget* mNodeFlow = nullptr;

		QAction* mNewFileAct;

		QAction* mOpenFileAct;

		QAction* mSaveFileAct;

		QAction* mSaveAsFileAct;

		QAction* mCloseAct;

		QAction* mCloseAllAct;

		QString mFileName;
	};
}