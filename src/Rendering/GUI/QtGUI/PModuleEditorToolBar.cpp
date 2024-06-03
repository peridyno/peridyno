#include "PModuleEditorToolBar.h"

#include <QLabel>
#include <QToolButton>

#include "Platform.h"

#include "ToolBar/ToolButtonStyle.h"
#include "ToolBar/CompactToolButton.h"

namespace dyno
{
	PModuleEditorToolBar::PModuleEditorToolBar(QWidget* parent) :
		QFrame(parent)
	{
		mLayout = new QHBoxLayout;

		mActionSave = this->addAction(QToolButton::InstantPopup, new QAction(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/File/Save.png")), tr("&Save...")));
		mActionUpdate = this->addAction(QToolButton::InstantPopup, new QAction(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/Node/refresh_green.png")), tr("&Update...")));
		mActionReorder = this->addAction(QToolButton::InstantPopup, new QAction(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/Node/realign_v2.png")), tr("&Realign...")));

		mLayout->addStretch();

		mResetButton = this->addPushButton(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/Node/refresh_blue.png")), "Reset");
		mAnimationButton = this->addPushButton(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/Node/animation.png")), "Animation");
		mRenderingButton = this->addPushButton(QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/Node/Display.png")), "Rendering");

		mResetButton->setObjectName("mResetButton");
		mAnimationButton->setObjectName("mAnimationButton");
		mRenderingButton->setObjectName("mRenderingButton");

		mResetButton->setChecked(false);
		mAnimationButton->setChecked(true);
		mRenderingButton->setChecked(false);

		this->setLayout(mLayout);

		connect(mResetButton, &QPushButton::released, this, &PModuleEditorToolBar::resetButtonClicked);
		connect(mAnimationButton, &QPushButton::released, this, &PModuleEditorToolBar::animationButtonClicked);
		connect(mRenderingButton, &QPushButton::released, this, &PModuleEditorToolBar::renderingButtonClicked);

		this->setStyleSheet("border-radius: 4px; border: 1px solid rgb(120,120,120);");

		mResetButton->setStyleSheet("border: none;");
		mRenderingButton->setStyleSheet("border: none;");
		mAnimationButton->setStyleSheet("border: none;");

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

	QPushButton* PModuleEditorToolBar::addPushButton(QPixmap icon, QString text)
	{
		const int iconSize = 48;
		QPushButton* button = new QPushButton;

		button->setIconSize(QSize(iconSize, iconSize));
		button->setFixedWidth(160);
		button->setIcon(icon);//直接调用PushButton的setIcon和setText添加icon和文字，以修复Label布局下文字无法高亮的问题。
		button->setText(text);//

		button->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);

		button->setCheckable(true);
/*
		QLabel* iconLabel = new QLabel;    
		QLabel* textLabel = new QLabel;

		//iconLabel->setStyleSheet("background: transparent;");
		//textLabel->setStyleSheet("background: transparent;");

		iconLabel->resize(iconSize, iconSize);         
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
*/
		mLayout->addWidget(button);

		return button;
	}

	void PModuleEditorToolBar::resetButtonClicked()
	{
		mResetButton->setChecked(true);
		mAnimationButton->setChecked(false);
		mRenderingButton->setChecked(false);

		emit showResetPipeline();
	}

	void PModuleEditorToolBar::animationButtonClicked()
	{
		mResetButton->setChecked(false);
		mAnimationButton->setChecked(true);
		mRenderingButton->setChecked(false);

		emit showAnimationPipeline();
	}

	void PModuleEditorToolBar::renderingButtonClicked()
	{
		mResetButton->setChecked(false);
		mAnimationButton->setChecked(false);
		mRenderingButton->setChecked(true);

		emit showGraphicsPipeline();
	}
}

