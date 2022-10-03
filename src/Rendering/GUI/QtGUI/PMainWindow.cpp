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
#include "ToolBar/ToolBarPage.h"
#include "Platform.h"

#include "PMainToolBar.h"
#include "PModuleEditorToolBar.h"

namespace dyno
{
	Q_DECLARE_METATYPE(QDockWidget::DockWidgetFeatures)

	PMainWindow::PMainWindow(
		RenderEngine* engine, 
		QWidget *parent, Qt::WindowFlags flags)
		: QMainWindow(parent, flags),
		m_statusBar(nullptr),
		m_propertyWidget(nullptr),
		m_animationWidget(nullptr)
	{
		setObjectName("MainWindow");
		setWindowTitle(QString("PeriDyno Studio ") + QString::number(PERIDYNO_VERSION_MAJOR) + QString(".") + QString::number(PERIDYNO_VERSION_MINOR) + QString(".") + QString::number(PERIDYNO_VERSION_PATCH) + QString(":  An AI-targeted physical simulation platform"));
		setWindowIcon(QIcon(QString::fromStdString(getAssetPath() + "logo/logo2.png")));


		mOpenGLWidget = new POpenGLWidget(engine);
		setCentralView();


		setupStatusBar();
//		setupMenuBar();
		setupAllWidgets();

		setupToolBar();

		connect(mToolBar, &PMainToolBar::nodeCreated, mNodeFlowView->node_scene, &Qt::QtNodeFlowScene::dynoNodePlaced);
		connect(mToolBar, &PMainToolBar::nodeCreated, PSimulationThread::instance(), &PSimulationThread::resetNode);

		connect(m_propertyWidget, &PPropertyWidget::nodeUpdated, PSimulationThread::instance(), &PSimulationThread::resetNode);
		connect(PSimulationThread::instance(), &PSimulationThread::oneFrameFinished, mOpenGLWidget, &POpenGLWidget::updateGrpahicsContext);
		connect(PSimulationThread::instance(), &PSimulationThread::sceneGraphChanged, mNodeFlowView->node_scene, &Qt::QtNodeFlowScene::updateNodeGraphView);

		connect(mToolBar, &PMainToolBar::logActTriggered, mIoDockerWidget, &PIODockWidget::toggleLogging);

		statusBar()->showMessage(tr("Status Bar"));
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

 	//	QWidget* viewWidget = new QWidget();
 	//	QHBoxLayout* hLayout = new QHBoxLayout();
	//	viewWidget->setLayout(hLayout);
 	//	hLayout->addWidget(m_vtkOpenglWidget, 1);
 	//	hLayout->addWidget(m_flowView, 1);

 		mainLayout->addWidget(m_animationWidget, 0);
		
	}

	void PMainWindow::showAboutMsg()

	{
		QMessageBox msgBox(this);

		msgBox.setWindowTitle("About");

		msgBox.setTextFormat(Qt::RichText);   //this is what makes the links clickable

		msgBox.setText("this is diagWindows");

		msgBox.setIconPixmap(QPixmap(":/ico/res/ExcelReport.ico"));

		msgBox.exec();

	}

	void PMainWindow::addNodeByName(std::string name) {
		mNodeFlowView->node_scene->addNodeByString(name);
	}

	void PMainWindow::setupToolBar()
	{
		mToolBar = new PMainToolBar(mNodeFlowView, this, 61, 3);
		mToolBar->setWindowTitle("Tool Bar");

		addToolBar(Qt::TopToolBarArea, mToolBar);
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
		QMessageBox::about(this, tr("PeriDyno Studio "), versoin);
		return;
	}

	void PMainWindow::showModuleEditor()
	{
		auto nodes = mNodeFlowView->node_scene->selectedNodes();
		Qt::QtNodeWidget* clickedNode = nullptr;
		if (nodes.size() > 0) {
			clickedNode = dynamic_cast<Qt::QtNodeWidget*>(nodes[0]->nodeDataModel());
		}

		if (clickedNode == nullptr)
			return;
		
		PModuleEditor* moduelEditor = new PModuleEditor(clickedNode);
		moduelEditor->setWindowTitle("Module Flow Editor");
		moduelEditor->resize(1024, 600);
		moduelEditor->setMinimumSize(512, 360);

		moduelEditor->setWindowModality(Qt::ApplicationModal);
		moduelEditor->setAttribute(Qt::WA_ShowModal, true);
		moduelEditor->setAttribute(Qt::WA_DeleteOnClose, true);
		moduelEditor->show();

		connect(moduelEditor, &PModuleEditor::changed, mOpenGLWidget, &POpenGLWidget::updateGraphicsContext);
		connect(moduelEditor->moduleFlowScene(), &Qt::QtModuleFlowScene::nodeExportChanged, mNodeFlowView->node_scene, &Qt::QtNodeFlowScene::updateNodeGraphView);
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
			{ "Property", 0, Qt::RightDockWidgetArea },
			{ "NodeEditor", 0, Qt::RightDockWidgetArea },
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
		propertyDockWidget->setMinimumWidth(580);
		addDockWidget(sets[2].area, propertyDockWidget);
		m_propertyWidget = new PPropertyWidget();
		propertyDockWidget->setWidget(m_propertyWidget);
		
		mIoDockerWidget = new PIODockWidget(this, Qt::WindowFlags(sets[1].flags));
		mIoDockerWidget->setWindowIcon(qtIcon);
		addDockWidget(sets[1].area, mIoDockerWidget);
		//windowMenu->addMenu(bottomDockWidget->colorSwatchMenu());

		setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
		setCorner(Qt::BottomRightCorner, Qt::RightDockWidgetArea);

		connect(mNodeFlowView->node_scene, &Qt::QtNodeFlowScene::nodeSelected, m_propertyWidget, &PPropertyWidget::showNodeProperty);
//		connect(m_moduleFlowView->module_scene, &QtNodes::QtModuleFlowScene::nodeSelected, m_propertyWidget, &PPropertyWidget::showBlockProperty);

		connect(mNodeFlowView->node_scene, &Qt::QtNodeFlowScene::nodeDoubleClicked, this, &PMainWindow::showModuleEditor);

		connect(m_propertyWidget, &PPropertyWidget::stateFieldUpdated, mNodeFlowView->node_scene, &Qt::QtNodeFlowScene::fieldUpdated);
	}

	void PMainWindow::mousePressEvent(QMouseEvent *event)
	{
		// 	QLichtThread* m_thread = new QLichtThread(openGLWidget->winId());
		// 	m_thread->start();

		
	}

}