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

		QMenu* fileMenu = new QMenu("File");

		auto newAction = fileMenu->addAction("New..");
		auto saveAction = fileMenu->addAction("Save..");
		auto loadAction = fileMenu->addAction("Load..");
		auto clearAction = fileMenu->addAction("Clear..");

		menuBar->addMenu(fileMenu);

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
	}

	QtModuleFlowWidget::~QtModuleFlowWidget()
	{
	}
}