#include "PSettingEditor.h"

#include <QHBoxLayout>
#include <QDebug>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPainter>
#include <QPushButton>
#include <QToolButton>
#include <QSvgRenderer>
#include "qgridlayout.h"

#include "PDockWidget.h"
#include "PModuleEditorToolBar.h"

#include "ToolBar/Group.h"
#include "ToolBar/ToolButtonStyle.h"
#include "ToolBar/CompactToolButton.h"

#include "PPropertyWidget.h"
#include "qscrollarea.h"

namespace dyno
{
	PSettingEditor::PSettingEditor(std::shared_ptr<SceneGraph> scn, QWidget* widget)
		: QMainWindow(nullptr)
		
	{
		//this->setFixedSize(800,600);
		this->setWindowTitle(QString("Setting"));
		this->setSceneGraph(scn); 
		//Set up property dock widget

		QDockWidget* Docker = new QDockWidget();
		this->addDockWidget(Qt::LeftDockWidgetArea, Docker);
		Docker->setMinimumSize(QSize(200, 600));
		Docker->setMaximumWidth(200);
		auto titleBar1 = Docker->titleBarWidget();
		Docker->setTitleBarWidget(new QWidget());
		delete titleBar1;

		QLabel* SelectLabel = new QLabel("Cateory");
		SelectLabel->setAlignment(Qt::AlignCenter);
		PPushButton* SceneSettingbtr = new PPushButton("SceneSetting");
		PPushButton* OtherSettingbtr = new PPushButton("Other");
		QWidget* selectWidget = new QWidget();
		QVBoxLayout* selectVLayout = new QVBoxLayout();

		settingWidget = new PSceneSetting(this, "Scene Setting");
		settingWidget->updateData();

		this->connect(SceneSettingbtr, SIGNAL(active(QString)), this, SLOT(buildSceneSettingWidget()));
		this->connect(OtherSettingbtr, SIGNAL(active(QString)), this, SLOT(buildOtherSettingWidget()));

		selectVLayout->addWidget(SelectLabel);
		selectVLayout->addWidget(SceneSettingbtr);
		selectVLayout->addWidget(OtherSettingbtr);
		selectVLayout->addStretch();

		selectWidget->setLayout(selectVLayout);
		Docker->setWidget(selectWidget);

		DockerRight = new QDockWidget();
		this->addDockWidget(Qt::RightDockWidgetArea, DockerRight);
		auto titleBar2 = DockerRight->titleBarWidget();
		DockerRight->setMinimumSize(QSize(500,600));
		DockerRight->setTitleBarWidget(new QWidget());
		delete titleBar2;
		DockerRight->setWidget(settingWidget);
		

	}

	void PSettingEditor::buildSceneSettingWidget()
	{	
		if (settingWidget != nullptr)
		{
			delete settingWidget;
			settingWidget = nullptr;
		}

		settingWidget = new PSceneSetting(this,"Scene Setting");
		settingWidget->updateData();
		DockerRight->setWidget(settingWidget);	
	}

	void PSettingEditor::buildOtherSettingWidget()
	{
		if (settingWidget != nullptr) 
		{
			delete settingWidget;
			settingWidget = nullptr;
		}
			
		settingWidget = new POtherSetting(this,"Other");
		settingWidget->updateData();
		DockerRight->setWidget(settingWidget);
	}


	PSettingWidget::PSettingWidget(PSettingEditor* editor,std::string title)
	{
		Editor = editor;

		layoutRight = new QGridLayout;

		mMainLayout = new QVBoxLayout;
		mScrollArea = new QScrollArea;

		mMainLayout->setContentsMargins(0, 0, 0, 0);
		mMainLayout->setSpacing(0);

		mTitle = new QLabel(title.c_str());
		int detailLabelHeight = 40;
		mTitle->setMinimumHeight(detailLabelHeight);
		mTitle->setMaximumHeight(detailLabelHeight);
		mTitle->setAlignment(Qt::AlignCenter);
		mMainLayout->addWidget(mTitle);

		mMainLayout->addWidget(mScrollArea);



		mScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		mScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		mScrollArea->setWidgetResizable(true);

		mScrollLayout = new QGridLayout;
		mScrollLayout->setAlignment(Qt::AlignTop);
		mScrollLayout->setContentsMargins(0, 0, 0, 0);

		QWidget* m_scroll_widget = new QWidget;
		m_scroll_widget->setLayout(mScrollLayout);
		mScrollArea->setWidget(m_scroll_widget);
		mScrollArea->setContentsMargins(0, 0, 0, 0);

		setLayout(mMainLayout);
	};

	void PSettingWidget::updateData()
	{	

	}




}
