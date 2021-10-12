#include "PNodeEditor.h"

#include <QHBoxLayout>

#include "PDockWidget.h"

#include "PPropertyWidget.h"
#include "PModuleFlowWidget.h"

namespace dyno
{
	PNodeEditor::PNodeEditor(QtNodes::QtNodeWidget* node_widget)
		: QMainWindow(nullptr, 0)
	{
		PModuleFlowWidget* moduleFlowView = new PModuleFlowWidget();
		this->setCentralWidget(moduleFlowView);

		//Set up property dock widget
		PDockWidget *propertyDockWidget = new PDockWidget(tr("Property"), this, Qt::WindowFlags(0));
		propertyDockWidget->setWindowTitle("Module Property");
		this->addDockWidget(Qt::LeftDockWidgetArea, propertyDockWidget);
		
		PPropertyWidget* propertyWidget = new PPropertyWidget();
		propertyDockWidget->setWidget(propertyWidget);
		propertyDockWidget->setMinimumWidth(400);


		if (node_widget != nullptr)
		{
			moduleFlowView->getModuleFlowScene()->showNodeFlow(node_widget->getNode().get());
		}

		connect(moduleFlowView->module_scene, &QtNodes::QtModuleFlowScene::nodeSelected, propertyWidget, &PPropertyWidget::showBlockProperty);
	}
}
