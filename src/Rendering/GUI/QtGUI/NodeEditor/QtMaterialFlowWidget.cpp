#include "QtMaterialFlowWidget.h"

//QT
#include <QGridLayout>
#include <QVBoxLayout>
#include <QMenuBar>

//Qt Nodes
#include "nodes/QFlowView"
#include "nodes/QDataModelRegistry"

namespace Qt
{
	QtMaterialFlowWidget::QtMaterialFlowWidget(std::shared_ptr<dyno::CustomMaterial> src, QWidget* parent) :
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
		mMaterialFlow = new QtMaterialFlowScene(src,this);
		l->addWidget(new QtFlowView(mMaterialFlow));
		l->setContentsMargins(0, 0, 0, 0);
		l->setSpacing(0);

		QObject::connect(saveAction, &QAction::triggered,
			mMaterialFlow, &QtMaterialFlowScene::save);

		QObject::connect(loadAction, &QAction::triggered,
			mMaterialFlow, &QtMaterialFlowScene::load);

		QObject::connect(clearAction, &QAction::triggered,
			mMaterialFlow, &QtMaterialFlowScene::clearScene);
	}

	QtMaterialFlowWidget::~QtMaterialFlowWidget()
	{
	}
}