#include "PMaterialEditorToolBar.h"

#include <QLabel>
#include <QToolButton>

#include "Platform.h"

#include "ToolBar/ToolButtonStyle.h"
#include "ToolBar/CompactToolButton.h"

namespace dyno
{
	PMaterialEditorToolBar::PMaterialEditorToolBar(QWidget* parent) :
		QFrame(parent)
	{
		mLayout = new QHBoxLayout;

		mActionSave = this->addAction(QToolButton::InstantPopup, new QAction(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/File/Save.png")), tr("&Save...")));
		mActionUpdate = this->addAction(QToolButton::InstantPopup, new QAction(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/Node/refresh_green.png")), tr("&Update...")));
		mActionReorder = this->addAction(QToolButton::InstantPopup, new QAction(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/Node/realign_v2.png")), tr("&Realign...")));
		mActionRealtime = new QAction(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/Edit/Edit.png")), "RealTime");
		mActionRealtime->setCheckable(true);
		this->addAction(QToolButton::DelayedPopup, mActionRealtime);

		mLayout->addStretch();
		this->setLayout(mLayout);


		this->setStyleSheet("border-radius: 4px; border: 1px solid rgb(120,120,120);");

	}

	QAction* PMaterialEditorToolBar::addAction(QToolButton::ToolButtonPopupMode type, QAction* action, QMenu* menu /*= nullptr*/)
	{
		if (type == QToolButton::MenuButtonPopup)
		{
			mLayout->addWidget(new tt::CompactToolButton(action, menu, this));
		}
		else
		{
			const int iconSize = 48;
			QToolButton* btn = new QToolButton(this);
			btn->setProperty("TTInternal", QVariant(true));
			btn->setAutoRaise(true);
			btn->setDefaultAction(action);
			btn->setIconSize(QSize(iconSize, iconSize));
			btn->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
			btn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
			btn->setPopupMode(type);
			btn->setStyle(new tt::TTToolButtonStyle());
			if (menu)
				btn->setMenu(menu);
			mLayout->addWidget(btn);
		}

		return action;
	}


}

