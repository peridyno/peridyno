#pragma once

#include <QMainWindow>

#include "NodeEditor/QtNodeWidget.h"
#include "NodeEditor/QtModuleFlowScene.h"

#include "PPropertyWidget.h"
#include <QMouseEvent>
#include "qpushbutton.h"
#include "qgridlayout.h"



namespace dyno
{
	class PSettingWidget;
	class QVector3FieldWidget;
	class SceneGraph;


	class PSettingEditor :
		public QMainWindow
	{
		Q_OBJECT
	public:
		PSettingEditor(QWidget* widget = nullptr);

		~PSettingEditor() {}
		
		PSettingWidget* getSettingWidget() { return settingWidget; }

	signals:
		void changed(SceneGraph* scn);

	public slots:

		void buildSceneSettingWidget();


		void buildOtherSettingWidget();


	private:

		PSettingWidget* settingWidget = nullptr;
		QDockWidget* DockerRight = nullptr;
	};

	class PSettingWidget :
		public QWidget
	{
		Q_OBJECT
	public:
		PSettingWidget(PSettingEditor* editor,std::string title);

		~PSettingWidget() {}

		QGridLayout* getScrollLayout() { return mScrollLayout; }
		PSettingEditor* getEditor() { return Editor; }
		void setLabelTitle(std::string text) { mTitle->setText(QString(text.c_str())); };

	signals:
		void changed(SceneGraph* scn);

	public slots:
		
		virtual void updateData() { ; }

	private:

		PSettingEditor* Editor = nullptr;
		QGridLayout* layoutRight = nullptr;
		QVBoxLayout* mMainLayout = nullptr;
		QScrollArea* mScrollArea = nullptr;
		QGridLayout* mScrollLayout = nullptr;
		QLabel* mTitle = nullptr;
	};



	class PSceneSetting :
		public PSettingWidget
	{
		Q_OBJECT
	public:
		PSceneSetting(PSettingEditor* editor,std::string title) 
			:PSettingWidget(editor,title)
		{
			;
		}


		~PSceneSetting() {}
	public slots:

		void updateData() override;

		void setGravity(double v0, double v1, double v2);
		void setLowerBound(double v0, double v1, double v2);
		void setUpperBound(double v0, double v1, double v2);


	private:

		QVector3FieldWidget* gravityWidget = nullptr;
		QVector3FieldWidget* lowerBoundWidget = nullptr;
		QVector3FieldWidget* upperBoundWidget = nullptr;


	};

	class POtherSetting :
		public PSettingWidget
	{
		Q_OBJECT
	public:
		POtherSetting(PSettingEditor* editor,std::string title) 
		:PSettingWidget(editor,title)
		{}

		~POtherSetting() {}
		 
	public slots:

		void updateData() override
		{
			QFont font("Microsoft YaHei", 20, 75);

			QLabel* overviveLabel = new QLabel("Overview");
			overviveLabel->setFont(font);

			QLabel* textLabel1 = new QLabel("PeriDyno is a CUDA-based, highly parallal physics engine targeted at providing real-time simulation of physical environments for intelligent agents.");
			textLabel1->setWordWrap(true);

			QLabel* licenseLabel = new QLabel("License");
			licenseLabel->setFont(font);
			

			QLabel* textLabel2 = new QLabel("Peridyno's default license is the Apache 2.0 (See LICENSE).\nExternal libraries are distributed under their own terms.\n");	
			textLabel2->setWordWrap(true);

			getScrollLayout()->addWidget(overviveLabel);
			getScrollLayout()->addWidget(textLabel1);
			getScrollLayout()->addWidget(licenseLabel);
			getScrollLayout()->addWidget(textLabel2);
		}

	};


	class PPushButton :
		public QPushButton
	{
		Q_OBJECT
	public:
		PPushButton(const QString& text, QWidget* parent = nullptr)
			: QPushButton(text, parent)
		{
			str = text;
		};

		~PPushButton() {}

	signals:
		void active(QString);

	public slots:

	protected:
		void mousePressEvent(QMouseEvent* e) override
		{
			emit active(str);
		}


	private:
		QString str = nullptr;


	};

}