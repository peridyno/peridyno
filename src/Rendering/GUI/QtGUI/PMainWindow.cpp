/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the demonstration applications of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/
#include "PMainWindow.h"
#include "PDockWidget.h"
#include "PToolBar.h"
#include "PStatusBar.h"
#include "POpenGLWidget.h"
#include "PAnimationWidget.h"

#include "PIODockWidget.h"
#include "PLogWidget.h"
#include "PConsoleWidget.h"
//#include "PSceneGraphWidget.h"
#include "PPropertyWidget.h"
//#include "PModuleListWidget.h"
#include "PSimulationThread.h"
#include "PModuleEditor.h"

#include "NodeEditor/QtNodeFlowWidget.h"
#include "NodeEditor/QtModuleFlowWidget.h"

#include <QAction>
#include <QLayout>
#include <QMenu>
#include <QMenuBar>
#include <QStatusBar>
#include <QTextEdit>
#include <QFile>
#include <QDataStream>
#include <QFileDialog>
#include <QDialogButtonBox>
#include <QMessageBox>
#include <QSignalMapper>
#include <QApplication>
#include <QPainter>
#include <QMouseEvent>
#include <QLineEdit>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QDebug>
#include <QtWidgets/QOpenGLWidget>
#include <QtSvg/QSvgRenderer>
#include <QHBoxLayout>
#include <QDialog>
#include <QVBoxLayout>

#include "NodeEditor/QtModuleFlowScene.h"

#include "nodes/QDataModelRegistry"

#include "Toolbar/TabToolbar.h"
#include "Toolbar/Page.h"
#include "Toolbar/Group.h"
#include "Toolbar/SubGroup.h"
#include "Toolbar/StyleTools.h"
#include "Toolbar/Builder.h"

#include "Platform.h"


// #include "Node/NodeData.hpp"
// #include "Node/FlowScene.hpp"
// #include "Node/FlowView.hpp"
// #include "Node/FlowViewStyle.hpp"
// #include "Node/ConnectionStyle.hpp"
// #include "Node/DataModelRegistry.hpp"

//#include "models.h"

namespace dyno
{
	Q_DECLARE_METATYPE(QDockWidget::DockWidgetFeatures)

	PMainWindow::PMainWindow(
		RenderEngine* engine, 
		QWidget *parent, Qt::WindowFlags flags)
		: QMainWindow(parent, flags),
		m_statusBar(nullptr),
		//m_vtkOpenglWidget(nullptr),
		m_propertyWidget(nullptr),
		m_animationWidget(nullptr)
// 		m_scenegraphWidget(nullptr),
// 		m_moduleListWidget(nullptr)
	{
		setObjectName("MainWindow");
		setWindowTitle(QString("PeriDyno Studio ") + QString::number(PERIDYNO_VERSION_MAJOR) + QString(".") + QString::number(PERIDYNO_VERSION_MINOR) + QString(".") + QString::number(PERIDYNO_VERSION_PATCH) + QString(":  An AI-targeted physics simulation platform"));
		setWindowIcon(QPixmap("../../data/logo3.png"));


		mOpenGLWidget = new POpenGLWidget(engine);
		setCentralView();

		setupToolBar();
		setupStatusBar();
//		setupMenuBar();
		setupAllWidgets();

//		connect(m_scenegraphWidget, &PSceneGraphWidget::notifyNodeDoubleClicked, m_moduleFlowView->getModuleFlowScene(), &QtNodes::QtModuleFlowScene::showNodeFlow);

		statusBar()->showMessage(tr("Status Bar"));

		Qt::ConnectionStyle::setConnectionStyle(
			R"(
			  {
				"ConnectionStyle": {
				  "UseDataDefinedColors": true
				}
			  }
			  )");
	}

	void PMainWindow::mainLoop()
	{
	}

	void PMainWindow::createWindow(int width, int height)
	{

	}

	void PMainWindow::newScene()
	{
		QMessageBox::StandardButton reply;

		reply = QMessageBox::question(this, "Save", "Do you want to save your changes?",
			QMessageBox::Ok | QMessageBox::Cancel);
	}

	void PMainWindow::setCentralView()
	{
		QWidget* centralWidget = new QWidget();
		setCentralWidget(centralWidget);

		centralWidget->setContentsMargins(0, 0, 0, 0);
		QVBoxLayout* mainLayout = new QVBoxLayout();
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setSpacing(0);
		centralWidget->setLayout(mainLayout);

		//Setup views
		QTabWidget* tabWidget = new QTabWidget();
		tabWidget->setObjectName(QStringLiteral("tabWidget"));
		tabWidget->setGeometry(QRect(140, 60, 361, 241));

// 		//VTK-based visualization widget
// 		m_vtkOpenglWidget = new PVTKOpenGLWidget();
// 		m_vtkOpenglWidget->setObjectName(QStringLiteral("tabView"));
// 		m_vtkOpenglWidget->layout()->setMargin(0);
// 		tabWidget->addTab(m_vtkOpenglWidget, QString());
// 		tabWidget->setTabText(tabWidget->indexOf(m_vtkOpenglWidget), QApplication::translate("MainWindow", "View", Q_NULLPTR));
// 
// 		connect(PSimulationThread::instance(), SIGNAL(oneFrameFinished()), m_vtkOpenglWidget, SLOT(prepareRenderingContex()));
// 		mainLayout->addWidget(tabWidget, 0, 0);


// 		m_moduleFlowView = new PModuleFlowWidget();
// 		m_moduleFlowView->setObjectName(QStringLiteral("tabEditor"));
// 		tabWidget->addTab(m_moduleFlowView, QString());
// 		tabWidget->setTabText(tabWidget->indexOf(m_moduleFlowView), QApplication::translate("MainWindow", "Module Editor", Q_NULLPTR));

		//OpenGL-based visualization widget
// 		mOpenGLWidget->setObjectName(QStringLiteral("tabView"));
// 		mOpenGLWidget->layout()->setMargin(0);
// 		tabWidget->addTab(mOpenGLWidget, QString());
// 		tabWidget->setTabText(tabWidget->indexOf(mOpenGLWidget), QApplication::translate("MainWindow", "View", Q_NULLPTR));
		mainLayout->addWidget(mOpenGLWidget, 1);

		//Setup animation widget
		m_animationWidget = new PAnimationWidget(this);
		m_animationWidget->layout()->setMargin(0);

		connect(PSimulationThread::instance(), SIGNAL(oneFrameFinished()), mOpenGLWidget, SLOT(updateGraphicsContext()));

// 		QWidget* viewWidget = new QWidget();
// 		QHBoxLayout* hLayout = new QHBoxLayout();
// 		viewWidget->setLayout(hLayout);
// 		hLayout->addWidget(m_vtkOpenglWidget, 1);
// 		hLayout->addWidget(m_flowView, 1);

 		mainLayout->addWidget(m_animationWidget, 0);
	}

	void PMainWindow::setupToolBar()
	{
		tt::TabToolbar* tt = new tt::TabToolbar(this, 55, 3);
		addToolBar(Qt::TopToolBarArea, tt);

		QString mediaDir = "../../data/icon/";

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


		tt::Page* filePage = tt->AddPage(QPixmap(mediaDir + "48px-Document-open.png"), "File");
	
		auto fg1 = filePage->AddGroup("");
		//New action
		QAction *actionNew = new QAction(QPixmap(mediaDir + "48px-Document-new.png"), "New...");
		fg1->AddAction(QToolButton::DelayedPopup, actionNew);
			
		//Open action
		QAction *actionOpen = new QAction(QPixmap(mediaDir + "48px-Document-open.png"), "Open");
		fg1->AddAction(QToolButton::DelayedPopup, actionOpen);
		//Save action
		QAction *actionSave = new QAction(QPixmap(mediaDir + "48px-Document-save.png"), "Save");
		fg1->AddAction(QToolButton::DelayedPopup, actionSave);
		//Save as action
		QAction *actionSaveAs = new QAction(QPixmap(mediaDir + "48px-Document-save-as.png"), "Save As");
		fg1->AddAction(QToolButton::DelayedPopup, actionSaveAs);

		tt::Page* editPage = tt->AddPage(QPixmap(mediaDir + "48px-Preferences-system.png"), "Edit");
		tt::Group* e1 = editPage->AddGroup("Group 1");
		QAction *actionSetting = new QAction(QPixmap(mediaDir + "48px-Preferences-system.png"), "Settings");
		e1->AddAction(QToolButton::DelayedPopup, actionSetting);

// 		PToolBar *tb = new PToolBar(tr("Tool Bar"), this);
// 		toolBars.append(tb);
// 		addToolBar(tb);

		tt::Page* particlePage = tt->AddPage(convertIcon(mediaDir + "dyverso/icon-emi-fill.svg"), "Particle System ");
		auto pg1 = particlePage->AddGroup("");
		QAction *particle1 = new QAction(convertIcon(mediaDir + "dyverso/icon-emi-fill.svg"), "Particle 1");
		pg1->AddAction(QToolButton::DelayedPopup, particle1);

		QAction *particle2 = new QAction(convertIcon(mediaDir + "dyverso/icon-emi-bitmap.svg"), "Particle 2");
		pg1->AddAction(QToolButton::DelayedPopup, particle2);

		QAction *particle3 = new QAction(convertIcon(mediaDir + "dyverso/icon-emi-circle.svg"), "Particle 3");
		pg1->AddAction(QToolButton::DelayedPopup, particle3);


		tt::Page* heightPage = tt->AddPage(convertIcon(mediaDir + "icon-realwave.svg"), "Height Field ");
		auto hg1 = heightPage->AddGroup("");
		QAction *wave1 = new QAction(convertIcon(mediaDir + "icon-realwave.svg"), "Wave 1");
		hg1->AddAction(QToolButton::DelayedPopup, wave1);

		QAction *wave2 = new QAction(convertIcon(mediaDir + "icon-realwave-cresplash.svg"), "Wave 2");
		hg1->AddAction(QToolButton::DelayedPopup, wave2);

		QAction *wave3 = new QAction(convertIcon(mediaDir + "icon-realwave-objspash.svg"), "Wave 3");
		hg1->AddAction(QToolButton::DelayedPopup, wave3);


		//Finite element
		tt::Page* femPage = tt->AddPage(convertIcon(mediaDir + "daemon/icon-demon-vortex.svg"), "Finite Element ");
		auto femg1 = femPage->AddGroup("");
		QAction *soft1 = new QAction(convertIcon(mediaDir + "daemon/icon-demon-vortex.svg"), "Soft Body 1");
		femg1->AddAction(QToolButton::DelayedPopup, soft1);

		QAction *soft2 = new QAction(convertIcon(mediaDir + "daemon/icon-demon-heater.svg"), "Soft Body 2");
		femg1->AddAction(QToolButton::DelayedPopup, soft2);

		QAction *soft3 = new QAction(convertIcon(mediaDir + "daemon/icon-demon-ellipsoid.svg"), "Soft Body 3");
		femg1->AddAction(QToolButton::DelayedPopup, soft3);

		QAction *soft4 = new QAction(convertIcon(mediaDir + "daemon/icon-demon-stension.svg"), "Soft Body 4");
		femg1->AddAction(QToolButton::DelayedPopup, soft4);



		//Articulated rigids
		tt::Page* artPage = tt->AddPage(convertIcon(mediaDir + "geometry/icon-geometry-cube.svg"), "Rigid Body ");
		auto rigidg1 = artPage->AddGroup("");
		QAction *art1 = new QAction(convertIcon(mediaDir + "geometry/icon-geometry-cube.svg"), "Rigid 1");
		rigidg1->AddAction(QToolButton::DelayedPopup, art1);

		QAction *art2 = new QAction(convertIcon(mediaDir + "geometry/icon-geometry-cylinder.svg"), "Rigid 2");
		rigidg1->AddAction(QToolButton::DelayedPopup, art2);

		QAction *art3 = new QAction(convertIcon(mediaDir + "geometry/icon-geometry-rocket.svg"), "Rigid 3");
		rigidg1->AddAction(QToolButton::DelayedPopup, art3);

		QAction *art4 = new QAction(convertIcon(mediaDir + "geometry/icon-geometry-multibody.svg"), "Rigid 4");
		rigidg1->AddAction(QToolButton::DelayedPopup, art4);


		tt::Page* helpPage = tt->AddPage(QPixmap(mediaDir + "Help-browser.png"), "Help");
		auto helpg1 = helpPage->AddGroup("");
		QAction *help1 = new QAction(QPixmap(mediaDir + "Help-browser.png"), "Help");
		helpg1->AddAction(QToolButton::DelayedPopup, help1);
	}

	void PMainWindow::setupStatusBar()
	{
		m_statusBar = new PStatusBar(this);
		setStatusBar(m_statusBar);
	}

/*	void PMainWindow::setupMenuBar()
	{
		QMenu *menu = menuBar()->addMenu(tr("&File"));

		menu->addAction(tr("New ..."), this, &PMainWindow::newScene);
		menu->addAction(tr("Load ..."), this, &PMainWindow::loadScene);
		menu->addAction(tr("Save ..."), this, &PMainWindow::saveScene);

		menu->addSeparator();
		menu->addAction(tr("&Quit"), this, &QWidget::close);

		mainWindowMenu = menuBar()->addMenu(tr("&View"));
		mainWindowMenu->addAction(tr("FullScreen"), this, &PMainWindow::fullScreen);

#ifdef Q_OS_OSX
		toolBarMenu->addSeparator();

		action = toolBarMenu->addAction(tr("Unified"));
		action->setCheckable(true);
		action->setChecked(unifiedTitleAndToolBarOnMac());
		connect(action, &QAction::toggled, this, &QMainWindow::setUnifiedTitleAndToolBarOnMac);
#endif

		windowMenu = menuBar()->addMenu(tr("&Window"));
		for (int i = 0; i < toolBars.count(); ++i)
			windowMenu->addMenu(toolBars.at(i)->toolbarMenu());

		aboutMenu = menuBar()->addMenu(tr("&Help"));
		aboutMenu->addAction(tr("Show Help ..."), this, &PMainWindow::showHelp);
		aboutMenu->addAction(tr("About ..."), this, &PMainWindow::showAbout);
	}*/

	void PMainWindow::saveScene()
	{
		return;
	}

	void PMainWindow::fullScreen()
	{
		return;
	}

	void PMainWindow::showHelp()
	{
		return;
	}

	void PMainWindow::showAbout()
	{
		QString versoin = QString("Version ") + QString::number(PERIDYNO_VERSION_MAJOR)+QString(".")+ QString::number(PERIDYNO_VERSION_MINOR)+QString(".")+QString::number(PERIDYNO_VERSION_PATCH);
		QMessageBox::about(this, tr("PhysIKA Studio "), versoin);
		return;
	}

	void PMainWindow::showNodeEditor()
	{
		auto nodes = mNodeFlowView->node_scene->selectedNodes();
		Qt::QtNodeWidget* clickedBlock = nullptr;
		if (nodes.size() > 0)
		{
			clickedBlock = dynamic_cast<Qt::QtNodeWidget*>(nodes[0]->nodeDataModel());
		}
		
		PModuleEditor* node_editor = new PModuleEditor(clickedBlock);
		node_editor->setWindowTitle("Module Flow Editor");
		node_editor->resize(1024, 600);
		node_editor->setMinimumSize(512, 360);

		node_editor->setWindowModality(Qt::ApplicationModal);
		node_editor->setAttribute(Qt::WA_ShowModal, true);
		node_editor->setAttribute(Qt::WA_DeleteOnClose, true);
		node_editor->show();
	}

	void PMainWindow::showMessage()
	{
		std::cout << "ShowMessage" << std::endl;
	}

	void PMainWindow::loadScene()
	{
		return;
	}

	void PMainWindow::setupAllWidgets()
	{
		qRegisterMetaType<QDockWidget::DockWidgetFeatures>();

		//windowMenu->addSeparator();

		static const struct Set {
			const char * name;
			uint flags;
			Qt::DockWidgetArea area;
		} sets[] = {
			{ "SceneGraph", 0, Qt::RightDockWidgetArea },
			{ "Console", 0, Qt::BottomDockWidgetArea },
			{ "Property", 0, Qt::LeftDockWidgetArea },
			{ "NodeEditor", 0, Qt::LeftDockWidgetArea },
			{ "Module", 0, Qt::RightDockWidgetArea }
		};
		const int setCount = sizeof(sets) / sizeof(Set);

		const QIcon qtIcon(QPixmap(":/res/qt.png"));


		PDockWidget *nodeEditorDockWidget = new PDockWidget(tr(sets[3].name), this, Qt::WindowFlags(sets[3].flags));
		nodeEditorDockWidget->setWindowTitle("Node Editor");
		nodeEditorDockWidget->setWindowIcon(qtIcon);
		addDockWidget(sets[3].area, nodeEditorDockWidget);
		mNodeFlowView = new Qt::QtNodeFlowWidget();
		mNodeFlowView->setObjectName(QStringLiteral("tabEditor"));
		nodeEditorDockWidget->setWidget(mNodeFlowView);



		//Set up property dock widget
		PDockWidget *propertyDockWidget = new PDockWidget(tr(sets[2].name), this, Qt::WindowFlags(sets[2].flags));
		propertyDockWidget->setWindowTitle("Property Editor");
		propertyDockWidget->setWindowIcon(qtIcon);
		addDockWidget(sets[2].area, propertyDockWidget);
		//windowMenu->addMenu(moduleListDockWidget->colorSwatchMenu());
		m_propertyWidget = new PPropertyWidget();
//		m_propertyWidget->setOpenGLWidget(m_vtkOpenglWidget);
		propertyDockWidget->setWidget(m_propertyWidget);


		PIODockWidget *consoleDockWidget = new PIODockWidget(this, Qt::WindowFlags(sets[1].flags));
		consoleDockWidget->setWindowIcon(qtIcon);
		addDockWidget(sets[1].area, consoleDockWidget);
		//windowMenu->addMenu(bottomDockWidget->colorSwatchMenu());

		//Set up module widget
// 		PDockWidget *moduleDockWidget = new PDockWidget(tr(sets[4].name), this, Qt::WindowFlags(sets[4].flags));
// 		moduleDockWidget->setWindowTitle("Module List");
// 		moduleDockWidget->setWindowIcon(qtIcon);
// 		addDockWidget(sets[4].area, moduleDockWidget);
// 		//windowMenu->addMenu(rightDockWidget->colorSwatchMenu());
// 		m_moduleListWidget = new PModuleListWidget();
// 		moduleDockWidget->setWidget(m_moduleListWidget);
// 
// 		PDockWidget *sceneDockWidget = new PDockWidget(tr(sets[0].name), this, Qt::WindowFlags(sets[0].flags));
// 		sceneDockWidget->setWindowTitle("Scene Browser");
// 		sceneDockWidget->setWindowIcon(qtIcon);
// 		addDockWidget(sets[0].area, sceneDockWidget);
// 		//windowMenu->addMenu(leftDockWidget->colorSwatchMenu());
// 		m_scenegraphWidget = new PSceneGraphWidget();
// 		sceneDockWidget->setWidget(m_scenegraphWidget);

		setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
		setCorner(Qt::BottomRightCorner, Qt::RightDockWidgetArea);

// 		connect(m_scenegraphWidget, SIGNAL(notifyNodeSelected(Node*)), m_moduleListWidget, SLOT(updateModule(Node*)));
// 		connect(m_scenegraphWidget, SIGNAL(notifyNodeSelected(Node*)), m_propertyWidget, SLOT(showProperty(Node*)));
// 		connect(m_moduleListWidget, SIGNAL(notifyModuleSelected(Module*)), m_propertyWidget, SLOT(showProperty(Module*)));

		connect(mNodeFlowView->node_scene, &Qt::QtNodeFlowScene::nodeSelected, m_propertyWidget, &PPropertyWidget::showBlockProperty);
//		connect(m_moduleFlowView->module_scene, &QtNodes::QtModuleFlowScene::nodeSelected, m_propertyWidget, &PPropertyWidget::showBlockProperty);

		connect(mNodeFlowView->node_scene, &Qt::QtNodeFlowScene::nodeDoubleClicked, this, &PMainWindow::showNodeEditor);
	}

	void PMainWindow::mousePressEvent(QMouseEvent *event)
	{
		// 	QLichtThread* m_thread = new QLichtThread(openGLWidget->winId());
		// 	m_thread->start();
	}

}