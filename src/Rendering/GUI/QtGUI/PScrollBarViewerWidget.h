#pragma once
#include "Field.h"
#include "Topology/TriangleSet.h"
#include "Topology/PolygonSet.h"

#include <QTableWidgetItem>
#include <QResizeEvent>
#include <QWheelEvent>
#include <QScrollBar>

#include <QScrollBar>
#include <QMouseEvent>
#include <QHBoxLayout>
#include <QKeyEvent>
#include "Format.h"

#include "ViewerItem/PDataViewerWidget.h"
#include "ViewerItem/PVec3FieldViewerWidget.h"

#include "ViewerItem/PIntegerViewerWidget.h"
#include "ViewerItem/PTransform3fViewerWidget.h"
#include "ViewerItem/PRealViewerWidget.h"
#include "ViewerItem/PTriangleSetViewerWidget.h"
#include "ViewerItem/PPointSetViewerWidget.h"
#include "ViewerItem/PPolygonSetViewerWidget.h"
#include "ViewerItem/PTextureMeshViewerWidget.h"
#include "ViewerItem/PDiscreteElementViewerWidget.h"

#include "PSimulationThread.h"
#include <QCloseEvent>



namespace dyno
{
	struct Field_Type 
	{
		Field_Type(std::shared_ptr<FBase> ptr, std::string type)
		{
			mPtr = ptr;
			mType = type;
		}
	
		std::shared_ptr<FBase> mPtr;
		std::string mType = "";
	};


	class PScrollBarViewerWidget : public QWidget
	{
		Q_OBJECT

	public:
		PScrollBarViewerWidget(FBase* field,QWidget* parent = nullptr) : QWidget(parent)
		{
			mfield = field;
			//set Title
			this->setWindowTitle(FormatFieldWidgetName(field->getObjectName()) + " ( " + FormatFieldWidgetName(field->getTemplateName()) + " )");
			Qt::WindowFlags m_flags = windowFlags();
			setWindowFlags(m_flags | Qt::WindowStaysOnTopHint);

			QHBoxLayout* layout = new QHBoxLayout();
			this->setLayout(layout);
			this->setAttribute(Qt::WA_DeleteOnClose, true);

			// initial widget
			// Vec3

			if (field->getClassName() == std::string("FVar") || field->getClassName() == std::string("FArray")) 
			{
				PDataViewerWidget* tableView = NULL;

				if (field->getTemplateName() == std::string(typeid(Vec3f).name()) || field->getTemplateName() == std::string(typeid(Vec3d).name())) 
					tableView = new PVec3FieldViewerWidget(field, NULL);

				// int / uint
				if (field->getTemplateName() == std::string(typeid(uint).name()) || field->getTemplateName() == std::string(typeid(int).name()))
					tableView = new PIntegerViewerWidget(field, NULL);

				// Transform3f
				if (field->getTemplateName() == std::string(typeid(Transform3f).name()))
					tableView = new PTransform3fViewerWidget(field, NULL);

				//// Real
				if (field->getTemplateName() == std::string(typeid(float).name()) || field->getTemplateName() == std::string(typeid(double).name()))
					tableView = new PRealViewerWidget(field, NULL);


				//construct

				//create ScrollBar
				QDataViewScrollBar* verticalScrollBar = new QDataViewScrollBar(Qt::Orientation::Vertical);

				if (tableView != NULL)
				{
					connect(PSimulationThread::instance(), &PSimulationThread::oneFrameFinished, tableView, &PDataViewerWidget::updateDataTable);
					connect(PSimulationThread::instance(), &PSimulationThread::sceneGraphChanged, tableView, &PDataViewerWidget::close);

					connect(verticalScrollBar, &QScrollBar::valueChanged, tableView, &PDataViewerWidget::updateDataTableTo);
					connect(tableView, &PDataViewerWidget::wheelDeltaAngleChange, verticalScrollBar, &QDataViewScrollBar::updateScrollValue);

					connect(PSimulationThread::instance(), &PSimulationThread::sceneGraphChanged, tableView, &PDataViewerWidget::close);


					tableView->setTableScrollBar(verticalScrollBar);


					layout->addWidget(tableView);
					layout->addWidget(verticalScrollBar);


					tableView->resize(maximumSize());
					verticalScrollBar->resize(verticalScrollBar->width(), this->height());

				}
				else
				{
					std::string info = "Unsupported Type : " + field->getTemplateName();//
					QLabel* errorInfo = new QLabel(QString::fromStdString(info), this);//FormatFieldWidgetName(field->getTemplateName())
					layout->addWidget(errorInfo);

				}

				this->show();
				return;
				
			}
			

			//************************* Instance **************************//
			{
				if (field->getClassName() == std::string("FInstance"))
				{
					//PointSet
					if (field->getTemplateName() == std::string(typeid(PointSet<DataType3f>).name())|| field->getTemplateName() == std::string(typeid(PointSet<DataType3d>).name()))
					{
						auto viewer = new PPointSetViewerWidget(field, NULL);
						layout->addWidget(viewer);
					}

					//TriangleSet
					if (field->getTemplateName() == std::string(typeid(TriangleSet<DataType3f>).name()) || field->getTemplateName() == std::string(typeid(TriangleSet<DataType3d>).name()))
					{
						auto viewer = new PTriangleSetViewerWidget(field, NULL);
						layout->addWidget(viewer);
					}
					//EdgeSet
					if (field->getTemplateName() == std::string(typeid(EdgeSet<DataType3f>).name()) || field->getTemplateName() == std::string(typeid(EdgeSet<DataType3d>).name()))
					{

					}
					//PolygonSet
					if (field->getTemplateName() == std::string(typeid(PolygonSet<DataType3f>).name()) || field->getTemplateName() == std::string(typeid(PolygonSet<DataType3d>).name()))
					{
						auto viewer = new PPolygonSetViewerWidget(field, NULL);
						layout->addWidget(viewer);
					}
					//TextureMesh
					if (field->getTemplateName() == std::string(typeid(TextureMesh).name()) || field->getTemplateName() == std::string(typeid(TextureMesh).name()))
					{
						auto viewer = new PTextureMeshViewerWidget(field, NULL);
						layout->addWidget(viewer);
					}
					//DiscreteElements
					if (field->getTemplateName() == std::string(typeid(DiscreteElements<DataType3f>).name()) || field->getTemplateName() == std::string(typeid(DiscreteElements<DataType3f>).name()))
					{
						auto viewer = new PDiscreteElementViewerWidget(field, NULL);
						layout->addWidget(viewer);
					}
				}
				else
				{
					std::string info = "Unsupported Type : " + field->getTemplateName();//
					QLabel* errorInfo = new QLabel(QString::fromStdString(info), this);//FormatFieldWidgetName(field->getTemplateName())
					layout->addWidget(errorInfo);
				}

				this->show();
				return;
			}
			
	

		}


		

			
	Q_SIGNALS:

				
		void updateInstanceWidgets();

	public slots:

		void updateInstanceField() 
		{
			emit updateInstanceWidgets();
		}
	

	protected:

		void resizeEvent(QResizeEvent* event)override
		{
			QWidget::resizeEvent(event);
		}



		void closeEvent(QCloseEvent* event) override
		{

			event->accept(); 
		}



	private:

		FBase* mfield = NULL;

	};

	
	
}
