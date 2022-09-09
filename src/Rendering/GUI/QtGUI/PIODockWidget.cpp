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

		setupActions();
		setupMenu();
		setupWidgets();
	}

	void PIODockWidget::updateContextMenu()
	{
		const Qt::DockWidgetArea area = mainWindow->dockWidgetArea(this);
		const Qt::DockWidgetAreas areas = allowedAreas();

		closableAction->setChecked(features() & QDockWidget::DockWidgetClosable);
		if (windowType() == Qt::Drawer) {
			floatableAction->setEnabled(false);
			floatingAction->setEnabled(false);
			movableAction->setEnabled(false);
			verticalTitleBarAction->setChecked(false);
		}
		else {
			floatableAction->setChecked(features() & QDockWidget::DockWidgetFloatable);
			floatingAction->setChecked(isWindow());
			// done after floating, to get 'floatable' correctly initialized
			movableAction->setChecked(features() & QDockWidget::DockWidgetMovable);
			verticalTitleBarAction
				->setChecked(features() & QDockWidget::DockWidgetVerticalTitleBar);
		}

		allowLeftAction->setChecked(isAreaAllowed(Qt::LeftDockWidgetArea));
		allowRightAction->setChecked(isAreaAllowed(Qt::RightDockWidgetArea));
		allowTopAction->setChecked(isAreaAllowed(Qt::TopDockWidgetArea));
		allowBottomAction->setChecked(isAreaAllowed(Qt::BottomDockWidgetArea));

		if (allowedAreasActions->isEnabled()) {
			allowLeftAction->setEnabled(area != Qt::LeftDockWidgetArea);
			allowRightAction->setEnabled(area != Qt::RightDockWidgetArea);
			allowTopAction->setEnabled(area != Qt::TopDockWidgetArea);
			allowBottomAction->setEnabled(area != Qt::BottomDockWidgetArea);
		}

		{
			const QSignalBlocker blocker(leftAction);
			leftAction->setChecked(area == Qt::LeftDockWidgetArea);
		}
		{
			const QSignalBlocker blocker(rightAction);
			rightAction->setChecked(area == Qt::RightDockWidgetArea);
		}
		{
			const QSignalBlocker blocker(topAction);
			topAction->setChecked(area == Qt::TopDockWidgetArea);
		}
		{
			const QSignalBlocker blocker(bottomAction);
			bottomAction->setChecked(area == Qt::BottomDockWidgetArea);
		}

		if (areaActions->isEnabled()) {
			leftAction->setEnabled(areas & Qt::LeftDockWidgetArea);
			rightAction->setEnabled(areas & Qt::RightDockWidgetArea);
			topAction->setEnabled(areas & Qt::TopDockWidgetArea);
			bottomAction->setEnabled(areas & Qt::BottomDockWidgetArea);
		}

		tabMenu->clear();
		splitHMenu->clear();
		splitVMenu->clear();
		QList<PIODockWidget*> dock_list = mainWindow->findChildren<PIODockWidget*>();
		foreach(PIODockWidget *dock, dock_list) {
			tabMenu->addAction(dock->objectName());
			splitHMenu->addAction(dock->objectName());
			splitVMenu->addAction(dock->objectName());
		}
	}

	static PIODockWidget *findByName(const QMainWindow *mainWindow, const QString &name)
	{
		foreach(PIODockWidget *dock, mainWindow->findChildren<PIODockWidget*>()) {
			if (name == dock->objectName())
				return dock;
		}
		return Q_NULLPTR;
	}

	void PIODockWidget::splitInto(QAction *action)
	{
		PIODockWidget *target = findByName(mainWindow, action->text());
		if (!target)
			return;

		const Qt::Orientation o = action->parent() == splitHMenu
			? Qt::Horizontal : Qt::Vertical;
		mainWindow->splitDockWidget(target, this, o);
	}

	void PIODockWidget::tabInto(QAction *action)
	{
		if (PIODockWidget *target = findByName(mainWindow, action->text()))
			mainWindow->tabifyDockWidget(target, this);
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
		menu->exec(event->globalPos());
	}
#endif // QT_NO_CONTEXTMENU

	void PIODockWidget::allow(Qt::DockWidgetArea area, bool a)
	{
		Qt::DockWidgetAreas areas = allowedAreas();
		areas = a ? areas | area : areas & ~area;
		setAllowedAreas(areas);

		if (areaActions->isEnabled()) {
			leftAction->setEnabled(areas & Qt::LeftDockWidgetArea);
			rightAction->setEnabled(areas & Qt::RightDockWidgetArea);
			topAction->setEnabled(areas & Qt::TopDockWidgetArea);
			bottomAction->setEnabled(areas & Qt::BottomDockWidgetArea);
		}
	}

	void PIODockWidget::place(Qt::DockWidgetArea area, bool p)
	{
		if (!p)
			return;

		mainWindow->addDockWidget(area, this);

		if (allowedAreasActions->isEnabled()) {
			allowLeftAction->setEnabled(area != Qt::LeftDockWidgetArea);
			allowRightAction->setEnabled(area != Qt::RightDockWidgetArea);
			allowTopAction->setEnabled(area != Qt::TopDockWidgetArea);
			allowBottomAction->setEnabled(area != Qt::BottomDockWidgetArea);
		}
	}

	void PIODockWidget::setupActions()
	{
		closableAction = new QAction(tr("Closable"), this);
		closableAction->setCheckable(true);
		connect(closableAction, &QAction::triggered, this, &PIODockWidget::changeClosable);

		movableAction = new QAction(tr("Movable"), this);
		movableAction->setCheckable(true);
		connect(movableAction, &QAction::triggered, this, &PIODockWidget::changeMovable);

		floatableAction = new QAction(tr("Floatable"), this);
		floatableAction->setCheckable(true);
		connect(floatableAction, &QAction::triggered, this, &PIODockWidget::changeFloatable);

		verticalTitleBarAction = new QAction(tr("Vertical title bar"), this);
		verticalTitleBarAction->setCheckable(true);
		connect(verticalTitleBarAction, &QAction::triggered,
			this, &PIODockWidget::changeVerticalTitleBar);

		floatingAction = new QAction(tr("Floating"), this);
		floatingAction->setCheckable(true);
		connect(floatingAction, &QAction::triggered, this, &PIODockWidget::changeFloating);

		allowedAreasActions = new QActionGroup(this);
		allowedAreasActions->setExclusive(false);

		allowLeftAction = new QAction(tr("Allow on Left"), this);
		allowLeftAction->setCheckable(true);
		connect(allowLeftAction, &QAction::triggered, this, &PIODockWidget::allowLeft);

		allowRightAction = new QAction(tr("Allow on Right"), this);
		allowRightAction->setCheckable(true);
		connect(allowRightAction, &QAction::triggered, this, &PIODockWidget::allowRight);

		allowTopAction = new QAction(tr("Allow on Top"), this);
		allowTopAction->setCheckable(true);
		connect(allowTopAction, &QAction::triggered, this, &PIODockWidget::allowTop);

		allowBottomAction = new QAction(tr("Allow on Bottom"), this);
		allowBottomAction->setCheckable(true);
		connect(allowBottomAction, &QAction::triggered, this, &PIODockWidget::allowBottom);

		allowedAreasActions->addAction(allowLeftAction);
		allowedAreasActions->addAction(allowRightAction);
		allowedAreasActions->addAction(allowTopAction);
		allowedAreasActions->addAction(allowBottomAction);

		areaActions = new QActionGroup(this);
		areaActions->setExclusive(true);

		leftAction = new QAction(tr("Place on Left"), this);
		leftAction->setCheckable(true);
		connect(leftAction, &QAction::triggered, this, &PIODockWidget::placeLeft);

		rightAction = new QAction(tr("Place on Right"), this);
		rightAction->setCheckable(true);
		connect(rightAction, &QAction::triggered, this, &PIODockWidget::placeRight);

		topAction = new QAction(tr("Place on Top"), this);
		topAction->setCheckable(true);
		connect(topAction, &QAction::triggered, this, &PIODockWidget::placeTop);

		bottomAction = new QAction(tr("Place on Bottom"), this);
		bottomAction->setCheckable(true);
		connect(bottomAction, &QAction::triggered, this, &PIODockWidget::placeBottom);

		areaActions->addAction(leftAction);
		areaActions->addAction(rightAction);
		areaActions->addAction(topAction);
		areaActions->addAction(bottomAction);

		connect(movableAction, &QAction::triggered, areaActions, &QActionGroup::setEnabled);

		connect(movableAction, &QAction::triggered, allowedAreasActions, &QActionGroup::setEnabled);

		connect(floatableAction, &QAction::triggered, floatingAction, &QAction::setEnabled);

		connect(floatingAction, &QAction::triggered, floatableAction, &QAction::setDisabled);
		connect(movableAction, &QAction::triggered, floatableAction, &QAction::setEnabled);

		tabMenu = new QMenu(this);
		tabMenu->setTitle(tr("Tab into"));
		connect(tabMenu, &QMenu::triggered, this, &PIODockWidget::tabInto);

		splitHMenu = new QMenu(this);
		splitHMenu->setTitle(tr("Split horizontally into"));
		connect(splitHMenu, &QMenu::triggered, this, &PIODockWidget::splitInto);

		splitVMenu = new QMenu(this);
		splitVMenu->setTitle(tr("Split vertically into"));
		connect(splitVMenu, &QMenu::triggered, this, &PIODockWidget::splitInto);

		QAction *windowModifiedAction = new QAction(tr("Modified"), this);
		windowModifiedAction->setCheckable(true);
		windowModifiedAction->setChecked(false);
		connect(windowModifiedAction, &QAction::toggled, this, &QWidget::setWindowModified);
	}

	void PIODockWidget::setupMenu()
	{
		menu = new QMenu(this);
		menu->addAction(toggleViewAction());
		menu->addAction(tr("Raise"), this, &QWidget::raise);
		//    menu->addAction(tr("Change Size Hints..."), swatch, &ColorDock::changeSizeHints);

		menu->addSeparator();
		menu->addAction(closableAction);
		menu->addAction(movableAction);
		menu->addAction(floatableAction);
		menu->addAction(floatingAction);
		menu->addAction(verticalTitleBarAction);
		menu->addSeparator();
		menu->addActions(allowedAreasActions->actions());
		menu->addSeparator();
		menu->addActions(areaActions->actions());
		menu->addSeparator();
		menu->addMenu(splitHMenu);
		menu->addMenu(splitVMenu);
		menu->addMenu(tabMenu);
		menu->addSeparator();
//		menu->addAction(windowModifiedAction);

		connect(menu, &QMenu::aboutToShow, this, &PIODockWidget::updateContextMenu);
	}

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
		mLogWidget = new PLogWidget();
		m_ioTabWidget->addTab(mLogWidget, QString("Log"));
		m_ioTabWidget->setTabText(m_ioTabWidget->indexOf(mLogWidget), QApplication::translate("MainWindow", "Log", Q_NULLPTR));

		mConsoleWidget = new PConsoleWidget();
		m_ioTabWidget->addTab(mConsoleWidget, QString("Console"));
		m_ioTabWidget->setTabText(m_ioTabWidget->indexOf(mConsoleWidget), QApplication::translate("MainWindow", "Console", Q_NULLPTR));
		this->setWidget(m_ioTabWidget);

		QObject::connect(m_ioTabWidget, SIGNAL(currentChanged(int)), this, SLOT(changeTab(int)));

		this->setWindowTitle("Log");
	}

	void PIODockWidget::setCustomSizeHint(const QSize &size)
	{
		//     if (ColorDock *dock = qobject_cast<ColorDock*>(widget()))
		//         dock->setCustomSizeHint(size);
	}

	void PIODockWidget::changeClosable(bool on)
	{
		setFeatures(on ? features() | DockWidgetClosable : features() & ~DockWidgetClosable);
	}

	void PIODockWidget::changeMovable(bool on)
	{
		setFeatures(on ? features() | DockWidgetMovable : features() & ~DockWidgetMovable);
	}

	void PIODockWidget::changeFloatable(bool on)
	{
		setFeatures(on ? features() | DockWidgetFloatable : features() & ~DockWidgetFloatable);
	}

	void PIODockWidget::changeFloating(bool floating)
	{
		setFloating(floating);
	}

	void PIODockWidget::allowLeft(bool a)
	{
		allow(Qt::LeftDockWidgetArea, a);
	}

	void PIODockWidget::allowRight(bool a)
	{
		allow(Qt::RightDockWidgetArea, a);
	}

	void PIODockWidget::allowTop(bool a)
	{
		allow(Qt::TopDockWidgetArea, a);
	}

	void PIODockWidget::allowBottom(bool a)
	{
		allow(Qt::BottomDockWidgetArea, a);
	}

	void PIODockWidget::placeLeft(bool p)
	{
		place(Qt::LeftDockWidgetArea, p);
	}

	void PIODockWidget::placeRight(bool p)
	{
		place(Qt::RightDockWidgetArea, p);
	}

	void PIODockWidget::placeTop(bool p)
	{
		place(Qt::TopDockWidgetArea, p);
	}

	void PIODockWidget::placeBottom(bool p)
	{
		place(Qt::BottomDockWidgetArea, p);
	}

	void PIODockWidget::changeVerticalTitleBar(bool on)
	{
		setFeatures(on ? features() | DockWidgetVerticalTitleBar
			: features() & ~DockWidgetVerticalTitleBar);
	}
}