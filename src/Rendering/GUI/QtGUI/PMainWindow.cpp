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
#include "PConsoleWidget.h"
#include "PPropertyWidget.h"
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
#if QT_VERSION >= QT_VERSION_CHECK(6,0,0)
	#include <QtOpenGLWidgets/QtOpenGLWidgets>
#else
	#include <QtWidgets/QOpenGLWidget>
#endif
#include <QtSvg/QSvgRenderer>
#include <QHBoxLayout>
#include <QDialog>
#include <QVBoxLayout>

#include "NodeEditor/QtModuleFlowScene.h"
#include "NodeEditor/QtNodeWidget.h"

#include "nodes/QDataModelRegistry"
#include "nodes/QNode"
#include "ToolBar/TabToolbar.h"
#include "ToolBar/Page.h"
#include "ToolBar/Group.h"
#include "ToolBar/SubGroup.h"
#include "ToolBar/StyleTools.h"
#include "ToolBar/Builder.h"
#include "ToolBar/ToolBarPage.h"
#include "Platform.h"

#include "PMainToolBar.h"
#include "PModuleEditorToolBar.h"
#include "PSettingEditor.h"
#include "SceneGraphFactory.h"


namespace dyno
{
	//Q_DECLARE_METATYPE(QDockWidget::DockWidgetFeatures)

	PMainWindow::PMainWindow(
		QtApp* app,
		QWidget *parent, Qt::WindowFlags flags)
		: QMainWindow(parent, flags),
		mStatusBar(nullptr),
		mPropertyWidget(nullptr),
		mAnimationWidget(nullptr)
	{
		setObjectName("MainWindow");
		setWindowTitle(QString("PeriDyno Studio ") + QString::number(PERIDYNO_VERSION_MAJOR) + QString(".") + QString::number(PERIDYNO_VERSION_MINOR) + QString(".") + QString::number(PERIDYNO_VERSION_PATCH) + QString(":  An AI-targeted physical simulation platform"));
		setWindowIcon(QIcon(QString::fromStdString(getAssetPath() + "logo/logo5.png")));

		setCentralView();


		setupStatusBar();
//		setupMenuBar();
		setupAllWidgets();

		setupToolBar();

		setupSettingEditor();

		connect(mToolBar, &PMainToolBar::nodeCreated, mNodeFlowView->flowScene(), &Qt::QtNodeFlowScene::createQtNode);

		connect(mNodeFlowView->flowScene(), &Qt::QtNodeFlowScene::nodePlaced, PSimulationThread::instance(), &PSimulationThread::resetQtNode);

		connect(PSimulationThread::instance(), &PSimulationThread::oneFrameFinished, mOpenGLWidget, &POpenGLWidget::updateGrpahicsContext);
		connect(PSimulationThread::instance(), &PSimulationThread::sceneGraphChanged, mNodeFlowView->flowScene(), &Qt::QtNodeFlowScene::updateNodeGraphView);

		connect(mPropertyWidget, &PPropertyWidget::nodeUpdated, PSimulationThread::instance(), &PSimulationThread::syncNode);

		connect(mToolBar, &PMainToolBar::logActTriggered, mIoDockerWidget, &PIODockWidget::toggleLogging);
		
		connect(mToolBar, &PMainToolBar::settingTriggered, this, &PMainWindow::showSettingEditor);

		connect(this, &PMainWindow::updateSetting, mSettingEditor->getSettingWidget(), &PSettingWidget::updateData);

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
		
		mOpenGLWidget = new POpenGLWidget(this);
		mOpenGLWidget->setContentsMargins(0, 0, 0, 0);
		mainLayout->addWidget(mOpenGLWidget, 1);
		
		//Setup animation widget
		mAnimationWidget = new PAnimationWidget(this);
		mAnimationWidget->layout()->setContentsMargins(0, 0, 0, 0);

 		mainLayout->addWidget(mAnimationWidget, 0);
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
		mNodeFlowView->flowScene()->addNodeByString(name);
	}

	void PMainWindow::setupToolBar()
	{
		mToolBar = new PMainToolBar(mNodeFlowView, this, 61, 3);
		mToolBar->setWindowTitle("Tool Bar");

		addToolBar(Qt::TopToolBarArea, mToolBar);
	}

	void PMainWindow::setupStatusBar()
	{
		mStatusBar = new PStatusBar(this);
		setStatusBar(mStatusBar);
	}

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

	void PMainWindow::showModuleEditor(Qt::QtNode& s)
	{
		Qt::QtNodeWidget* clickedNode = nullptr;

		clickedNode = dynamic_cast<Qt::QtNodeWidget*>(s.nodeDataModel());

		if (clickedNode == nullptr)
			return;

		QString caption = s.nodeDataModel()->caption();

		PModuleEditor* moduelEditor = new PModuleEditor(clickedNode);
		moduelEditor->setWindowTitle("Module Flow Editor -- " + caption);
		moduelEditor->resize(1024, 600);
		moduelEditor->setMinimumSize(512, 360);

		moduelEditor->setWindowModality(Qt::ApplicationModal);
		moduelEditor->setAttribute(Qt::WA_ShowModal, true);
		moduelEditor->setAttribute(Qt::WA_DeleteOnClose, true);
		moduelEditor->show();

		connect(moduelEditor, &PModuleEditor::changed, mOpenGLWidget, &POpenGLWidget::updateGraphicsContext);
		connect(moduelEditor->moduleFlowScene(), &Qt::QtModuleFlowScene::nodeExportChanged, mNodeFlowView->flowScene(), &Qt::QtNodeFlowScene::updateNodeGraphView);
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
		mPropertyWidget = new PPropertyWidget();
		propertyDockWidget->setWidget(mPropertyWidget);
		
		mIoDockerWidget = new PIODockWidget(this, Qt::WindowFlags(sets[1].flags));
		mIoDockerWidget->setWindowIcon(qtIcon);
		addDockWidget(sets[1].area, mIoDockerWidget);
		//windowMenu->addMenu(bottomDockWidget->colorSwatchMenu());

		setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
		setCorner(Qt::BottomRightCorner, Qt::RightDockWidgetArea);

		connect(mNodeFlowView->flowScene(), &Qt::QtNodeFlowScene::nodeSelected, mPropertyWidget, &PPropertyWidget::showProperty);
//		connect(m_moduleFlowView->module_scene, &QtNodes::QtModuleFlowScene::nodeSelected, m_propertyWidget, &PPropertyWidget::showBlockProperty);

		connect(mNodeFlowView->flowScene(), &Qt::QtNodeFlowScene::nodeDoubleClicked, this, &PMainWindow::showModuleEditor);

		connect(mPropertyWidget, &PPropertyWidget::stateFieldUpdated, mNodeFlowView->flowScene(), &Qt::QtNodeFlowScene::fieldUpdated);

		// between OpenGL and property widget
		connect(mOpenGLWidget, &POpenGLWidget::nodeSelected, [=](std::shared_ptr<Node> node) {
			mPropertyWidget->showNodeProperty(node);
			// TODO: high light selected node in node editor
			auto qNodes = mNodeFlowView->flowScene()->allNodes();
			for (auto qNode : qNodes)
			{
				if (dynamic_cast<Qt::QtNodeWidget*>(qNode->nodeDataModel())->getNode() == node)
					qNode->nodeGraphicsObject().setSelected(true);
				else
					qNode->nodeGraphicsObject().setSelected(false);
			}
			});


		connect(mNodeFlowView->flowScene(), &Qt::QtNodeFlowScene::nodeSelected, [=](Qt::QtNode& n)
			{
				auto model = n.nodeDataModel();
				auto widget = dynamic_cast<Qt::QtNodeWidget*>(model);

				if (widget != nullptr)
				{
					mOpenGLWidget->select(widget->getNode());
					mOpenGLWidget->update();
				}
			});

		connect(mAnimationWidget, &PAnimationWidget::simulationStarted, [=]()
			{
				mOpenGLWidget->setFocus();
				mOpenGLWidget->setSelection(false);
			});

		connect(mAnimationWidget, &PAnimationWidget::simulationStopped, [=]()
			{
				mOpenGLWidget->setSelection(true);
			});
	}

	void PMainWindow::mousePressEvent(QMouseEvent *event)
	{
	}

	void PMainWindow::setupSettingEditor()
	{
		auto scn = SceneGraphFactory::instance()->active();

		mSettingEditor = new PSettingEditor(nullptr);

	}

	void PMainWindow::showSettingEditor() 
	{	
		if (mSettingEditor == nullptr)
			return;

		updateSettingData();
		mSettingEditor->show();
	}

	void PMainWindow::updateSettingData() 
	{
		if (mSceneGraph == nullptr)
			return;

		emit updateSetting();
	}



}
