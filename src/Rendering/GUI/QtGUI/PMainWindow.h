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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QApplication>
#include <QMainWindow>
#include <memory>

#include "NodeEditor/QtNodeFlowWidget.h"
#include "NodeEditor/QtModuleFlowWidget.h"

#include "nodes/internal/QtNode.hpp"


QT_FORWARD_DECLARE_CLASS(QMenu)

// class Qt::PNodeFlowWidget;
// class Qt::PModuleFlowWidget;

namespace dyno
{
	class PToolBar;
	class PStatusBar;
	class POpenGLWidget;
	class PIODockWidget;
	//class PVTKOpenGLWidget;
	//class PSceneGraphWidget;
	class PPropertyWidget;
	class PAnimationWidget;
	class PModuleListWidget;
	class PModuleEditor;
	class PMainToolBar;

	class QtApp;
	class ImWidget;
	class PSettingEditor;

//	QT_FORWARD_DECLARE_CLASS(QLichtWidget)

	class PMainWindow : public QMainWindow
	{
		Q_OBJECT

	public:
		typedef QMap<QString, QSize> CustomSizeHintMap;

		explicit PMainWindow(QtApp* app,
			QWidget *parent = Q_NULLPTR,
			Qt::WindowFlags flags = Qt::WindowFlags());

		void mainLoop();
		void createWindow(int width, int height);

		POpenGLWidget* openglWidget() { return mOpenGLWidget; }

	public slots:
		//File menu
		void newScene();
		void loadScene();
		void saveScene();

		//View menu
		void fullScreen();

		//Help menu
		void showHelp();
		void showAbout();

		void showModuleEditor(Qt::QtNode& n);

		void showMessage();

		void addNodeByName(std::string name);

		void showAboutMsg();
		
		//Setting Editor

		void showSettingEditor();


	signals:
		void updateSetting ();

	private:
		void setCentralView();
		void setupToolBar();
		void setupStatusBar();
//		void setupMenuBar();
		void setupAllWidgets();
		void setupSettingEditor();
		void updateSettingData();

	protected:
		void mousePressEvent(QMouseEvent *event) override;

	private:
		QApplication* application;

		QList<PToolBar*> toolBars;
		QList<QDockWidget *> extraDockWidgets;
		QMenu *destroyDockWidgetMenu;

		Qt::QtNodeFlowWidget*		mNodeFlowView;
		
		PStatusBar*				mStatusBar;
		POpenGLWidget*			mOpenGLWidget;

		PPropertyWidget*		mPropertyWidget;
		PAnimationWidget*		mAnimationWidget;

		PMainToolBar*			mToolBar = nullptr;

		PIODockWidget* mIoDockerWidget = nullptr;

		PSettingEditor* mSettingEditor = nullptr;


	};

}

#endif // MAINWINDOW_H
