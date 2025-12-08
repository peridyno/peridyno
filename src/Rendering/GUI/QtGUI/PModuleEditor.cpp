#include "PModuleEditor.h"

#include <QHBoxLayout>
#include <QDebug>

#include <QHBoxLayout>
#include <QPainter>
#include <QPushButton>
#include <QToolButton>
#include <QSvgRenderer>

#include "PDockWidget.h"
#include "PModuleEditorToolBar.h"

#include "ToolBar/Group.h"
#include "ToolBar/ToolButtonStyle.h"
#include "ToolBar/CompactToolButton.h"

#include "NodeEditor/QtModuleFlowWidget.h"

#include "PPropertyWidget.h"
#include "POpenGLWidget.h"

namespace dyno
{
	PModuleEditor::PModuleEditor(Qt::QtNodeWidget* widget,POpenGLWidget* openGLWidget)
		: QMainWindow(nullptr)
	{
		this->mOpenGLWidget = openGLWidget;

		mToolBar = new PModuleEditorToolBar();

		//Set up property dock widget
		QDockWidget* toolBarDocker = new QDockWidget();
		this->addDockWidget(Qt::TopDockWidgetArea, toolBarDocker);
		auto titleBar = toolBarDocker->titleBarWidget();
		toolBarDocker->setFixedHeight(96);
		toolBarDocker->setTitleBarWidget(new QWidget());
		delete titleBar;

		toolBarDocker->setWidget(mToolBar);


		auto moduleFlowView = new Qt::QtModuleFlowWidget(nullptr, widget);

		mModuleFlowScene = moduleFlowView->getModuleFlowScene();
		this->setCentralWidget(moduleFlowView);

		//Set up property dock widget
		PDockWidget *propertyDockWidget = new PDockWidget(tr("Property"), this, Qt::WindowFlags(0));
		propertyDockWidget->setWindowTitle("Property");
		this->addDockWidget(Qt::LeftDockWidgetArea, propertyDockWidget);
		
		PPropertyWidget* propertyWidget = new PPropertyWidget();
		propertyDockWidget->setWidget(propertyWidget);
		propertyDockWidget->setMinimumWidth(480);

		connect(moduleFlowView->mModuleFlow, &Qt::QtModuleFlowScene::nodeSelected, propertyWidget, &PPropertyWidget::showProperty);
		connect(moduleFlowView->mModuleFlow, &Qt::QtModuleFlowScene::nodeDeselected, propertyWidget, &PPropertyWidget::clearProperty);

		connect(mToolBar, &PModuleEditorToolBar::showResetPipeline, moduleFlowView->mModuleFlow, &Qt::QtModuleFlowScene::showResetPipeline);
		connect(mToolBar, &PModuleEditorToolBar::showAnimationPipeline, moduleFlowView->mModuleFlow, &Qt::QtModuleFlowScene::showAnimationPipeline);
		connect(mToolBar, &PModuleEditorToolBar::showGraphicsPipeline, moduleFlowView->mModuleFlow, &Qt::QtModuleFlowScene::showGraphicsPipeline);
		connect(mToolBar->updateAction(), &QAction::triggered, moduleFlowView->mModuleFlow, &Qt::QtModuleFlowScene::reconstructActivePipeline);
		if (mOpenGLWidget) 
			connect(propertyWidget, &PPropertyWidget::moduleUpdated, mOpenGLWidget, &POpenGLWidget::onModuleUpdated);

		connect(mToolBar->updateAction(), &QAction::triggered, 
			[=]() {
				emit changed(widget->getNode().get());
			});

		connect(mToolBar->reorderAction(), &QAction::triggered, moduleFlowView->mModuleFlow, &Qt::QtModuleFlowScene::reorderAllModules);
	}
}
