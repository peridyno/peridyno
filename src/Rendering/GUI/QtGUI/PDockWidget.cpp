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

#include "PDockWidget.h"

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

#undef DEBUG_SIZEHINTS

namespace dyno
{

	QColor bgColorForName(const QString &name)
	{
		if (name == "Black")
			return QColor("#D8D8D8");
		if (name == "White")
			return QColor("#F1F1F1");
		if (name == "Red")
			return QColor("#F1D8D8");
		if (name == "Green")
			return QColor("#D8E4D8");
		if (name == "Blue")
			return QColor("#D8D8F1");
		if (name == "Yellow")
			return QColor("#F1F0D8");
		return QColor(name).light(110);
	}

	QColor fgColorForName(const QString &name)
	{
		if (name == "Black")
			return QColor("#6C6C6C");
		if (name == "White")
			return QColor("#F8F8F8");
		if (name == "Red")
			return QColor("#F86C6C");
		if (name == "Green")
			return QColor("#6CB26C");
		if (name == "Blue")
			return QColor("#6C6CF8");
		if (name == "Yellow")
			return QColor("#F8F76C");
		return QColor(name);
	}

	PDockWidget::PDockWidget(const QString &colorName, QMainWindow *parent, Qt::WindowFlags flags) : 
		QDockWidget(parent, flags), 
		mainWindow(parent)
	{
		setObjectName(colorName + QLatin1String(" Dock Widget"));
		setWindowTitle(objectName() + QLatin1String(" [*]"));

		closableAction = new QAction(tr("Closable"), this);
		closableAction->setCheckable(true);
		connect(closableAction, &QAction::triggered, this, &PDockWidget::changeClosable);

		movableAction = new QAction(tr("Movable"), this);
		movableAction->setCheckable(true);
		connect(movableAction, &QAction::triggered, this, &PDockWidget::changeMovable);

		floatableAction = new QAction(tr("Floatable"), this);
		floatableAction->setCheckable(true);
		connect(floatableAction, &QAction::triggered, this, &PDockWidget::changeFloatable);

		verticalTitleBarAction = new QAction(tr("Vertical title bar"), this);
		verticalTitleBarAction->setCheckable(true);
		connect(verticalTitleBarAction, &QAction::triggered,
			this, &PDockWidget::changeVerticalTitleBar);

		floatingAction = new QAction(tr("Floating"), this);
		floatingAction->setCheckable(true);
		connect(floatingAction, &QAction::triggered, this, &PDockWidget::changeFloating);

		allowedAreasActions = new QActionGroup(this);
		allowedAreasActions->setExclusive(false);

		allowLeftAction = new QAction(tr("Allow on Left"), this);
		allowLeftAction->setCheckable(true);
		connect(allowLeftAction, &QAction::triggered, this, &PDockWidget::allowLeft);

		allowRightAction = new QAction(tr("Allow on Right"), this);
		allowRightAction->setCheckable(true);
		connect(allowRightAction, &QAction::triggered, this, &PDockWidget::allowRight);

		allowTopAction = new QAction(tr("Allow on Top"), this);
		allowTopAction->setCheckable(true);
		connect(allowTopAction, &QAction::triggered, this, &PDockWidget::allowTop);

		allowBottomAction = new QAction(tr("Allow on Bottom"), this);
		allowBottomAction->setCheckable(true);
		connect(allowBottomAction, &QAction::triggered, this, &PDockWidget::allowBottom);

		allowedAreasActions->addAction(allowLeftAction);
		allowedAreasActions->addAction(allowRightAction);
		allowedAreasActions->addAction(allowTopAction);
		allowedAreasActions->addAction(allowBottomAction);

		areaActions = new QActionGroup(this);
		areaActions->setExclusive(true);

		leftAction = new QAction(tr("Place on Left"), this);
		leftAction->setCheckable(true);
		connect(leftAction, &QAction::triggered, this, &PDockWidget::placeLeft);

		rightAction = new QAction(tr("Place on Right"), this);
		rightAction->setCheckable(true);
		connect(rightAction, &QAction::triggered, this, &PDockWidget::placeRight);

		topAction = new QAction(tr("Place on Top"), this);
		topAction->setCheckable(true);
		connect(topAction, &QAction::triggered, this, &PDockWidget::placeTop);

		bottomAction = new QAction(tr("Place on Bottom"), this);
		bottomAction->setCheckable(true);
		connect(bottomAction, &QAction::triggered, this, &PDockWidget::placeBottom);

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
		connect(tabMenu, &QMenu::triggered, this, &PDockWidget::tabInto);

		splitHMenu = new QMenu(this);
		splitHMenu->setTitle(tr("Split horizontally into"));
		connect(splitHMenu, &QMenu::triggered, this, &PDockWidget::splitInto);

		splitVMenu = new QMenu(this);
		splitVMenu->setTitle(tr("Split vertically into"));
		connect(splitVMenu, &QMenu::triggered, this, &PDockWidget::splitInto);

		QAction *windowModifiedAction = new QAction(tr("Modified"), this);
		windowModifiedAction->setCheckable(true);
		windowModifiedAction->setChecked(false);
		connect(windowModifiedAction, &QAction::toggled, this, &QWidget::setWindowModified);

		menu = new QMenu(colorName, this);
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
		menu->addAction(windowModifiedAction);

		connect(menu, &QMenu::aboutToShow, this, &PDockWidget::updateContextMenu);

		if (colorName == QLatin1String("Black")) {
			leftAction->setShortcut(Qt::CTRL | Qt::Key_W);
			rightAction->setShortcut(Qt::CTRL | Qt::Key_E);
			toggleViewAction()->setShortcut(Qt::CTRL | Qt::Key_R);
		}
	}

	void PDockWidget::updateContextMenu()
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
		QList<PDockWidget*> dock_list = mainWindow->findChildren<PDockWidget*>();
		foreach(PDockWidget *dock, dock_list) {
			tabMenu->addAction(dock->objectName());
			splitHMenu->addAction(dock->objectName());
			splitVMenu->addAction(dock->objectName());
		}
	}

	static PDockWidget *findByName(const QMainWindow *mainWindow, const QString &name)
	{
		foreach(PDockWidget *dock, mainWindow->findChildren<PDockWidget*>()) {
			if (name == dock->objectName())
				return dock;
		}
		return Q_NULLPTR;
	}

	void PDockWidget::splitInto(QAction *action)
	{
		PDockWidget *target = findByName(mainWindow, action->text());
		if (!target)
			return;

		const Qt::Orientation o = action->parent() == splitHMenu
			? Qt::Horizontal : Qt::Vertical;
		mainWindow->splitDockWidget(target, this, o);
	}

	void PDockWidget::tabInto(QAction *action)
	{
		if (PDockWidget *target = findByName(mainWindow, action->text()))
			mainWindow->tabifyDockWidget(target, this);
	}

#ifndef QT_NO_CONTEXTMENU
	void PDockWidget::contextMenuEvent(QContextMenuEvent *event)
	{
		event->accept();
		menu->exec(event->globalPos());
	}
#endif // QT_NO_CONTEXTMENU

	void PDockWidget::resizeEvent(QResizeEvent *e)
	{
		if (BlueTitleBar *btb = qobject_cast<BlueTitleBar*>(titleBarWidget()))
			btb->updateMask();

		QDockWidget::resizeEvent(e);
	}

	void PDockWidget::allow(Qt::DockWidgetArea area, bool a)
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

	void PDockWidget::place(Qt::DockWidgetArea area, bool p)
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

	void PDockWidget::setCustomSizeHint(const QSize &size)
	{
		//     if (ColorDock *dock = qobject_cast<ColorDock*>(widget()))
		//         dock->setCustomSizeHint(size);
	}

	void PDockWidget::changeClosable(bool on)
	{
		setFeatures(on ? features() | DockWidgetClosable : features() & ~DockWidgetClosable);
	}

	void PDockWidget::changeMovable(bool on)
	{
		setFeatures(on ? features() | DockWidgetMovable : features() & ~DockWidgetMovable);
	}

	void PDockWidget::changeFloatable(bool on)
	{
		setFeatures(on ? features() | DockWidgetFloatable : features() & ~DockWidgetFloatable);
	}

	void PDockWidget::changeFloating(bool floating)
	{
		setFloating(floating);
	}

	void PDockWidget::allowLeft(bool a)
	{
		allow(Qt::LeftDockWidgetArea, a);
	}

	void PDockWidget::allowRight(bool a)
	{
		allow(Qt::RightDockWidgetArea, a);
	}

	void PDockWidget::allowTop(bool a)
	{
		allow(Qt::TopDockWidgetArea, a);
	}

	void PDockWidget::allowBottom(bool a)
	{
		allow(Qt::BottomDockWidgetArea, a);
	}

	void PDockWidget::placeLeft(bool p)
	{
		place(Qt::LeftDockWidgetArea, p);
	}

	void PDockWidget::placeRight(bool p)
	{
		place(Qt::RightDockWidgetArea, p);
	}

	void PDockWidget::placeTop(bool p)
	{
		place(Qt::TopDockWidgetArea, p);
	}

	void PDockWidget::placeBottom(bool p)
	{
		place(Qt::BottomDockWidgetArea, p);
	}

	void PDockWidget::changeVerticalTitleBar(bool on)
	{
		setFeatures(on ? features() | DockWidgetVerticalTitleBar
			: features() & ~DockWidgetVerticalTitleBar);
	}

	QSize BlueTitleBar::minimumSizeHint() const
	{
		QDockWidget *dw = qobject_cast<QDockWidget*>(parentWidget());
		Q_ASSERT(dw != 0);
		QSize result(leftPm.width() + rightPm.width(), centerPm.height());
		if (dw->features() & QDockWidget::DockWidgetVerticalTitleBar)
			result.transpose();
		return result;
	}

	BlueTitleBar::BlueTitleBar(QWidget *parent)
		: QWidget(parent)
		, leftPm(QPixmap(":/res/titlebarLeft.png"))
		, centerPm(QPixmap(":/res/titlebarCenter.png"))
		, rightPm(QPixmap(":/res/titlebarRight.png"))
	{
	}

	void BlueTitleBar::paintEvent(QPaintEvent*)
	{
		QPainter painter(this);
		QRect rect = this->rect();

		QDockWidget *dw = qobject_cast<QDockWidget*>(parentWidget());
		Q_ASSERT(dw != 0);

		if (dw->features() & QDockWidget::DockWidgetVerticalTitleBar) {
			QSize s = rect.size();
			s.transpose();
			rect.setSize(s);

			painter.translate(rect.left(), rect.top() + rect.width());
			painter.rotate(-90);
			painter.translate(-rect.left(), -rect.top());
		}

		painter.drawPixmap(rect.topLeft(), leftPm);
		painter.drawPixmap(rect.topRight() - QPoint(rightPm.width() - 1, 0), rightPm);
		QBrush brush(centerPm);
		painter.fillRect(rect.left() + leftPm.width(), rect.top(),
			rect.width() - leftPm.width() - rightPm.width(),
			centerPm.height(), centerPm);
	}

	void BlueTitleBar::mouseReleaseEvent(QMouseEvent *event)
	{
		QPoint pos = event->pos();

		QRect rect = this->rect();

		QDockWidget *dw = qobject_cast<QDockWidget*>(parentWidget());
		Q_ASSERT(dw != 0);

		if (dw->features() & QDockWidget::DockWidgetVerticalTitleBar) {
			QPoint p = pos;
			pos.setX(rect.left() + rect.bottom() - p.y());
			pos.setY(rect.top() + p.x() - rect.left());

			QSize s = rect.size();
			s.transpose();
			rect.setSize(s);
		}

		const int buttonRight = 7;
		const int buttonWidth = 20;
		int right = rect.right() - pos.x();
		int button = (right - buttonRight) / buttonWidth;
		switch (button) {
		case 0:
			event->accept();
			dw->close();
			break;
		case 1:
			event->accept();
			dw->setFloating(!dw->isFloating());
			break;
		case 2: {
			event->accept();
			QDockWidget::DockWidgetFeatures features = dw->features();
			if (features & QDockWidget::DockWidgetVerticalTitleBar)
				features &= ~QDockWidget::DockWidgetVerticalTitleBar;
			else
				features |= QDockWidget::DockWidgetVerticalTitleBar;
			dw->setFeatures(features);
			break;
		}
		default:
			event->ignore();
			break;
		}
	}

	void BlueTitleBar::updateMask()
	{
		QDockWidget *dw = qobject_cast<QDockWidget*>(parent());
		Q_ASSERT(dw != 0);

		QRect rect = dw->rect();
		QPixmap bitmap(dw->size());

		{
			QPainter painter(&bitmap);

			// initialize to transparent
			painter.fillRect(rect, Qt::color0);

			QRect contents = rect;
			contents.setTopLeft(geometry().bottomLeft());
			contents.setRight(geometry().right());
			contents.setBottom(contents.bottom() - y());
			painter.fillRect(contents, Qt::color1);

			// let's paint the titlebar
			QRect titleRect = this->geometry();

			if (dw->features() & QDockWidget::DockWidgetVerticalTitleBar) {
				QSize s = rect.size();
				s.transpose();
				rect.setSize(s);

				QSize s2 = size();
				s2.transpose();
				titleRect.setSize(s2);

				painter.translate(rect.left(), rect.top() + rect.width());
				painter.rotate(-90);
				painter.translate(-rect.left(), -rect.top());
			}

			contents.setTopLeft(titleRect.bottomLeft());
			contents.setRight(titleRect.right());
			contents.setBottom(rect.bottom() - y());

			QRect rect = titleRect;

			painter.drawPixmap(rect.topLeft(), leftPm.mask());
			painter.fillRect(rect.left() + leftPm.width(), rect.top(),
				rect.width() - leftPm.width() - rightPm.width(),
				centerPm.height(), Qt::color1);
			painter.drawPixmap(rect.topRight() - QPoint(rightPm.width() - 1, 0), rightPm.mask());

			painter.fillRect(contents, Qt::color1);
		}

		dw->setMask(bitmap);
	}

}