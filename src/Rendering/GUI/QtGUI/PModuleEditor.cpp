#include "PModuleEditor.h"

#include <QHBoxLayout>
#include <QDebug>

#include "PDockWidget.h"

#include "PPropertyWidget.h"
#include "NodeEditor/QtModuleFlowWidget.h"


namespace dyno
{
	PModuleEditor::PModuleEditor(Qt::QtNodeWidget* node_widget)
		: QMainWindow(nullptr, 0)
	{
		Qt::QtModuleFlowWidget* moduleFlowView = new Qt::QtModuleFlowWidget(nullptr, node_widget);
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
			Node *selectedNode = node_widget->getNode().get();
			moduleFlowView->getModuleFlowScene()->showNodeFlow(selectedNode);

			// Here is Node's virtual module
			auto& scene = moduleFlowView->mModuleFlow;
			auto c =selectedNode->getClassInfo();
			auto type = scene->registry().create(QString::fromStdString(c->m_className + "(virtual)"));
			
			if (type)
			{
				auto& vir_module = scene->createNode(std::move(type));
				// Centered
				QPointF posView(120, 146);
				vir_module.nodeGraphicsObject().setPos(posView);
				scene->nodePlaced(vir_module);
			}
			else
			{
				qDebug() << "Model not found";
			}			
		}

		connect(moduleFlowView->mModuleFlow, &Qt::QtModuleFlowScene::nodeSelected, propertyWidget, &PPropertyWidget::showNodeProperty);
	}
}
