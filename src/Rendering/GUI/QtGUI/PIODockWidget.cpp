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

#include "PIODockWidget.h"
#include "PIOTabWidget.h"
#include "PLogWidget.h"
#include "PConsoleWidget.h"

#include <QAction>
#include <QtEvents>
#include <QFrame>
#include <QMainWindow>
#include <QMenu>
#include <QPainter>
#include <QImage>
#include <QColor>
#include <QDialog>
#include <QDialogButtonBox>
#include <QGridLayout>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QLabel>
#include <QPainterPath>
#include <QPushButton>
#include <QHBoxLayout>
#include <QBitmap>
#include <QtDebug>
#include <QApplication>

#undef DEBUG_SIZEHINTS

namespace dyno
{
	PIODockWidget::PIODockWidget(QMainWindow *parent, Qt::WindowFlags flags)
		: QDockWidget(parent, flags), 
		mainWindow(parent),
		m_ioTabWidget(nullptr)
	{
		setObjectName(QLatin1String("IO Dock Widget"));
		setWindowTitle(objectName() + QLatin1String(" [*]"));

		setupWidgets();
	}

	static PIODockWidget *findByName(const QMainWindow *mainWindow, const QString &name)
	{
		foreach(PIODockWidget *dock, mainWindow->findChildren<PIODockWidget*>()) {
			if (name == dock->objectName())
				return dock;
		}
		return Q_NULLPTR;
	}

	void PIODockWidget::changeTab(int index)
	{
		QWidget* currentWidget = m_ioTabWidget->currentWidget();

		this->setWindowTitle(m_ioTabWidget->tabText(index));
	}

	void PIODockWidget::toggleLogging()
	{
		mLogWidget->toggleLogging();
	}

#ifndef QT_NO_CONTEXTMENU
	void PIODockWidget::contextMenuEvent(QContextMenuEvent *event)
	{
		event->accept();
//		menu->exec(event->globalPos());
	}
#endif // QT_NO_CONTEXTMENU


	void PIODockWidget::setupWidgets()
	{
		m_ioTabWidget = new PIOTabWidget();
		m_ioTabWidget->setObjectName("ControlPanel");
		m_ioTabWidget->setContentsMargins(0, 0, 0, 0);
		m_ioTabWidget->setTabPosition(QTabWidget::South);
		m_ioTabWidget->setObjectName(QStringLiteral("tabWidget"));
		m_ioTabWidget->setGeometry(QRect(140, 60, 361, 241));

		m_ioTabWidget->tabBar()->setObjectName("ControlPanelTabBar");

		//Create log widget
		mLogWidget = PLogWidget::instance();
		m_ioTabWidget->addTab(mLogWidget, QString("Log"));
		m_ioTabWidget->setTabText(m_ioTabWidget->indexOf(mLogWidget), QApplication::translate("MainWindow", "Log", Q_NULLPTR));

		mConsoleWidget = new PConsoleWidget();
		m_ioTabWidget->addTab(mConsoleWidget, QString("Console"));
		m_ioTabWidget->setTabText(m_ioTabWidget->indexOf(mConsoleWidget), QApplication::translate("MainWindow", "Console", Q_NULLPTR));
		this->setWidget(m_ioTabWidget);

		QObject::connect(m_ioTabWidget, SIGNAL(currentChanged(int)), this, SLOT(changeTab(int)));

		this->setWindowTitle("Log");
	}
}