#include "QtModuleFlowWidget.h"

//QT
#include <QGridLayout>
#include <QVBoxLayout>
#include <QMenuBar>

//Qt Nodes
#include "nodes/QFlowView"
#include "nodes/QDataModelRegistry"

namespace Qt
{
	QtModuleFlowWidget::QtModuleFlowWidget(QWidget *parent, QtNodeWidget* node_widget) :
		QWidget(parent)
	{
		auto menuBar = new QMenuBar();
		auto newAction = menuBar->addAction("New..");
		auto saveAction = menuBar->addAction("Save..");
		auto loadAction = menuBar->addAction("Load..");
		auto clearAction = menuBar->addAction("Clear..");
		auto pushAction = menuBar->addAction("PushModule..");

		QVBoxLayout *l = new QVBoxLayout(this);

		l->addWidget(menuBar);
		module_scene = new QtModuleFlowScene(this, node_widget);
		l->addWidget(new QtFlowView(module_scene));
		l->setContentsMargins(0, 0, 0, 0);
		l->setSpacing(0);

		QObject::connect(saveAction, &QAction::triggered,
			module_scene, &QtModuleFlowScene::save);

		QObject::connect(loadAction, &QAction::triggered,
			module_scene, &QtModuleFlowScene::load);

		QObject::connect(clearAction, &QAction::triggered,
			module_scene, &QtModuleFlowScene::clearScene);

		QObject::connect(pushAction, &QAction::triggered,
			module_scene, &QtModuleFlowScene::pushModule);
	}

	QtModuleFlowWidget::~QtModuleFlowWidget()
	{
	}
}