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
		mModuleFlow = new QtModuleFlowScene(this, node_widget);
		l->addWidget(new QtFlowView(mModuleFlow));
		l->setContentsMargins(0, 0, 0, 0);
		l->setSpacing(0);

		QObject::connect(saveAction, &QAction::triggered,
			mModuleFlow, &QtModuleFlowScene::save);

		QObject::connect(loadAction, &QAction::triggered,
			mModuleFlow, &QtModuleFlowScene::load);

		QObject::connect(clearAction, &QAction::triggered,
			mModuleFlow, &QtModuleFlowScene::clearScene);

		QObject::connect(pushAction, &QAction::triggered,
			mModuleFlow, &QtModuleFlowScene::pushModule);
	}

	QtModuleFlowWidget::~QtModuleFlowWidget()
	{
	}
}