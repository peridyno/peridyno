#include "PModuleEditorToolBar.h"

#include <QLabel>
#include <QToolButton>

#include "Platform.h"

#include "ToolBar/ToolButtonStyle.h"
#include "ToolBar/CompactToolButton.h"

namespace dyno
{
	PModuleEditorToolBar::PModuleEditorToolBar(QWidget* parent) :
		QWidget(parent)
	{
		mLayout = new QHBoxLayout;

		mActionSave = this->addAction(QToolButton::InstantPopup, new QAction(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/File/Save.png")), tr("&Save...")));
		mActionUpdate = this->addAction(QToolButton::InstantPopup, new QAction(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/Node/refresh_green.png")), tr("&Update...")));
		mActionReorder = this->addAction(QToolButton::InstantPopup, new QAction(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/Node/realign_v2.png")), tr("&Realign...")));

		mLayout->addStretch();

		mAnimationButton = this->addPushButton(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/Node/animation.png")), "Animation");
		mRenderingButton = this->addPushButton(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/Node/Display.png")), "Rendering");

		mAnimationButton->setObjectName("mAnimationButton");
		mRenderingButton->setObjectName("mRenderingButton");

		this->setLayout(mLayout);

		connect(mAnimationButton, &QPushButton::released, this, &PModuleEditorToolBar::animationButtonClicked);
		connect(mRenderingButton, &QPushButton::released, this, &PModuleEditorToolBar::renderingButtonClicked);
	}

	QAction* PModuleEditorToolBar::addAction(QToolButton::ToolButtonPopupMode type, QAction* action, QMenu* menu /*= nullptr*/)
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

	QPushButton* PModuleEditorToolBar::addPushButton(QPixmap& icon, QString text)
	{
		const int iconSize = 48;
		QPushButton* button = new QPushButton;

		button->setIconSize(QSize(iconSize, iconSize));
		button->setFixedWidth(160);
		button->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);

		button->setCheckable(true);

		QLabel* iconLabel = new QLabel;    
		QLabel* textLabel = new QLabel;

		//iconLabel->setStyleSheet("background: transparent;");
		//textLabel->setStyleSheet("background: transparent;");

		iconLabel->resize(iconSize, iconSize);         //   Fix the bug that the icon was cropped. Change from "setFixedSize" to "resize"
		iconLabel->setPixmap(icon);
		textLabel->setText(text);

		iconLabel->setMinimumWidth(iconSize);
		textLabel->setMinimumWidth(60);
		//textLabel->setFixedWidth(60);    
		QHBoxLayout* btnLayout = new QHBoxLayout();
		btnLayout->setSpacing(0);

		btnLayout->addSpacing(2);    
		btnLayout->addWidget(iconLabel);    
		btnLayout->addSpacing(2);    
		btnLayout->addWidget(textLabel);
		btnLayout->addSpacing(2);
		button->setLayout(btnLayout);

// 		button->setStyleSheet(
// 			"QPushButton{border: 1px solid #dcdfe6; padding: 10px; border-radius: 5px; background-color: #ffffff;}"
// 			"QPushButton:hover{background-color: #ecf5ff; color: #409eff;}"
// 			"QPushButton:checked{border: 1px solid #3a8ee6; color: #409eff;}");

		mLayout->addWidget(button);

		return button;
	}

	void PModuleEditorToolBar::animationButtonClicked()
	{
		mAnimationButton->setChecked(true);
		mRenderingButton->setChecked(false);

		emit showAnimationPipeline();
	}

	void PModuleEditorToolBar::renderingButtonClicked()
	{
		mAnimationButton->setChecked(false);
		mRenderingButton->setChecked(true);

		emit showGraphicsPipeline();
	}
}

