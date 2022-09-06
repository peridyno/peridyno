#include "PMainToolBar.h"

#include <QPainter>
#include <QToolButton>
#include <QFileDialog>
#include <QtSvg/QSvgRenderer>

//tool bar
#include "ToolBar/Page.h"
#include "ToolBar/Group.h"
#include "ToolBar/ToolBarPage.h"

#include "PSimulationThread.h"

//Framework
#include "NodeFactory.h"
#include "SceneGraphFactory.h"
#include "SceneLoaderFactory.h"

//Core
#include "Platform.h"

namespace dyno
{
	PMainToolBar::PMainToolBar(Qt::QtNodeFlowWidget* nodeFlow, QWidget* parent, unsigned _groupMaxHeight, unsigned _groupRowCount) :
		tt::TabToolbar(parent, _groupMaxHeight, _groupRowCount)
	{
		mNodeFlow = nodeFlow;

		setupFileMenu();
		setupEditMenu();

		QString mediaDir = QString::fromLocal8Bit(getAssetPath().c_str()) + "icon/";

		auto convertIcon = [&](QString path) -> QIcon
		{
			QSvgRenderer svg_render(path);
			QPixmap pixmap(48, 48);
			pixmap.fill(Qt::transparent);
			QPainter painter(&pixmap);
			svg_render.render(&painter);
			QIcon ico(pixmap);

			return ico;
		};

		//Add ToolBar page
		ToolBarPage m_toolBarPage;
		std::vector<ToolBarIcoAndLabel> v_IcoAndLabel = m_toolBarPage.tbl;

		for (int i = 0; i < v_IcoAndLabel.size(); i++) {
			ToolBarIcoAndLabel m_tbl = v_IcoAndLabel[i];

			tt::Page* MainPage = this->AddPage(QPixmap(mediaDir + m_tbl.tabPageIco), m_tbl.tabPageName);
			auto m_page = MainPage->AddGroup("");

			for (int j = 0; j < m_tbl.ico.size(); j++) {
				//Add subtabs
				QAction* art = new QAction(QPixmap(mediaDir + m_tbl.ico[j]), m_tbl.label[j]);
				m_page->AddAction(QToolButton::DelayedPopup, art);

				if (i == 2 || i == 5 || i == 3) {//add connect event 
					//connect(art, &QAction::triggered, this, [=]() {addNodeByName(m_tbl.label[j].toStdString() + "<DataType3f>"); });
				}

				//reoder¹¦ÄÜ
				if (i == v_IcoAndLabel.size() - 1 && j == 0) {
					connect(art, &QAction::triggered, this, [=]() {mNodeFlow->node_scene->reorderAllNodes(); });
				}
			}
		}

		//Add dynamic toolbar page
		auto& pages = NodeFactory::instance()->nodePages();
		for(auto iPage = pages.begin(); iPage != pages.end(); iPage++)
		{
			auto& groups = iPage->second->groups();

			tt::Page* page = this->AddPage(QPixmap(mediaDir + QString::fromStdString(iPage->second->icon())), QString::fromStdString(iPage->second->caption()));

			for (auto iGroup = groups.begin(); iGroup != groups.end(); iGroup++)
			{
				auto& actions = iGroup->second->actions();

				auto qGroup = page->AddGroup("");
				for (size_t i = 0; i < actions.size(); i++)
				{
					QAction* act = new QAction(QPixmap(mediaDir + QString::fromStdString(actions[i]->icon())), QString::fromStdString(actions[i]->caption()));
					qGroup->AddAction(QToolButton::DelayedPopup, act);

					auto func = actions[i]->action();

					if (func != nullptr) {
						connect(act, &QAction::triggered, [=]() {
							emit nodeCreated(func());
							});
					}
				}
			}
		}
	}

	void PMainToolBar::newFile()
	{
		PSimulationThread::instance()->createNewScene();
		//emit newSceneLoaded();
	}

	void PMainToolBar::openFile()
	{
		mFileName = QFileDialog::getOpenFileName(this, tr("Open New ..."), "", tr("Xml Files (*.xml)"));
		if (!mFileName.isEmpty()) {
			auto scnLoader = SceneLoaderFactory::getInstance().getEntryByFileExtension("xml");
			auto scn = scnLoader->load(mFileName.toStdString());

			if (scn) {
				PSimulationThread::instance()->createNewScene(scn);
				//SceneGraphFactory::instance()->pushScene(scn);
			}
			//emit newSceneLoaded();
		}
	}

	void PMainToolBar::saveFile()
	{
		if (mFileName.isEmpty())
			mFileName = QFileDialog::getSaveFileName(this, tr("Save ..."), "", tr("Xml Files (*.xml)"));

		if (!mFileName.isEmpty())
		{
			auto scnLoader = SceneLoaderFactory::getInstance().getEntryByFileExtension("xml");
			scnLoader->save(SceneGraphFactory::instance()->active(), mFileName.toStdString());
		}
	}

	void PMainToolBar::saveAsFile()
	{
		mFileName = QFileDialog::getSaveFileName(this, tr("Save As ..."), "", tr("Xml Files (*.xml)"));
		if (!mFileName.isEmpty())
		{
			auto scnLoader = SceneLoaderFactory::getInstance().getEntryByFileExtension("xml");
			scnLoader->save(SceneGraphFactory::instance()->active(), mFileName.toStdString());
		}
	}

	void PMainToolBar::closeFile()
	{
		PSimulationThread::instance()->closeCurrentScene();
	}

	void PMainToolBar::closeAllFiles()
	{
		PSimulationThread::instance()->closeAllScenes();
	}

	void PMainToolBar::setupFileMenu()
	{
		QString path = QString::fromStdString(getAssetPath());
		tt::Page* filePage = this->AddPage(QPixmap(path + "icon/ToolBarIco/File/Open.png"), "File");

		auto fileGroup = filePage->AddGroup("File");

		mNewFileAct = new QAction(QPixmap(path + "icon/ToolBarIco/File/New_v2.png"), "New");
		fileGroup->AddAction(QToolButton::DelayedPopup, mNewFileAct);

		mOpenFileAct = new QAction(QPixmap(path + "icon/ToolBarIco/File/Open.png"), "Open");
		fileGroup->AddAction(QToolButton::DelayedPopup, mOpenFileAct);

		mSaveFileAct = new QAction(QPixmap(path + "icon/ToolBarIco/File/Save.png"), "Save");
		fileGroup->AddAction(QToolButton::DelayedPopup, mSaveFileAct);

		mSaveAsFileAct = new QAction(QPixmap(path + "icon/ToolBarIco/File/SaveAs.png"), "Save As");
		fileGroup->AddAction(QToolButton::DelayedPopup, mSaveAsFileAct);

		mCloseAct = new QAction(QPixmap(path + "icon/ToolBarIco/File/CloseSence.png"), "Close");
		fileGroup->AddAction(QToolButton::DelayedPopup, mCloseAct);

		mCloseAllAct = new QAction(QPixmap(path + "icon/ToolBarIco/File/CloseAll_v2.png"), "Close All");
		fileGroup->AddAction(QToolButton::DelayedPopup, mCloseAllAct);

		connect(mNewFileAct, &QAction::triggered, this, &PMainToolBar::newFile);
		connect(mOpenFileAct, &QAction::triggered, this, &PMainToolBar::openFile);
		connect(mSaveFileAct, &QAction::triggered, this, &PMainToolBar::saveFile);
		connect(mSaveAsFileAct, &QAction::triggered, this, &PMainToolBar::saveAsFile);
		connect(mCloseAct, &QAction::triggered, this, &PMainToolBar::closeFile);
		connect(mCloseAllAct, &QAction::triggered, this, &PMainToolBar::closeAllFiles);
	}

	void PMainToolBar::setupEditMenu()
	{
		QString path = QString::fromStdString(getAssetPath());
		tt::Page* filePage = this->AddPage(QPixmap(path + "icon/ToolBarIco/Edit/Settings_v2.png"), "Edit");

		auto configGroup = filePage->AddGroup("Config");

		mLogAct = new QAction(QPixmap(path + "icon/ToolBarIco/Edit/Edit.png"), "Log");
		configGroup->AddAction(QToolButton::DelayedPopup, mLogAct);

		mEditAct = new QAction(QPixmap(path + "icon/ToolBarIco/Edit/Settings_v2.png"), "Settings");
		configGroup->AddAction(QToolButton::DelayedPopup, mEditAct);

		connect(mLogAct, &QAction::triggered, this, [=]() { emit logActTriggered(); });
	}
}

