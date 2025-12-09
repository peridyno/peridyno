#pragma once
#include "Field.h"
#include "Topology/TriangleSet.h"
#include "Module/TopologyModule.h"

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

				QHBoxLayout* layout = new QHBoxLayout(this);
				//create ScrollBar
				QDataViewScrollBar* verticalScrollBar = new QDataViewScrollBar(Qt::Orientation::Vertical);

				if (tableView != NULL)
				{
					connect(PSimulationThread::instance(), &PSimulationThread::sceneGraphChanged, tableView, &PDataViewerWidget::close);

					connect(verticalScrollBar, &QScrollBar::valueChanged, tableView, &PDataViewerWidget::updateDataTableTo);
					connect(tableView, &PDataViewerWidget::wheelDeltaAngleChange, verticalScrollBar, &QDataViewScrollBar::updateScrollValue);

					connect(PSimulationThread::instance(), &PSimulationThread::sceneGraphChanged, tableView, &PDataViewerWidget::close);

					this->setAttribute(Qt::WA_DeleteOnClose, true);

					//verticalScrollBar = new QDataViewScrollBar(Qt::Vertical);

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
				QHBoxLayout* layout = new QHBoxLayout(this);
				if (field->getClassName() == std::string("FInstance"))
				{
					//PointSet
					if (field->getTemplateName() == std::string(typeid(PointSet<DataType3f>).name()))
					{
						updateInstanceData(field);

						std::vector<PDataViewerWidget*> viewers;
						for (auto it : Name_Field)
						{
							auto f = it.second;
							PDataViewerWidget* temp;
							if (f->mType == std::string(typeid(Vec3f).name()))
							{
								temp = new PVec3FieldViewerWidget(f->mPtr.get());
								viewers.push_back(temp);
							}

							QDataViewScrollBar* verticalScrollBar = new QDataViewScrollBar(Qt::Orientation::Vertical);
							layout->addWidget(temp);
							layout->addWidget(verticalScrollBar);

							connect(verticalScrollBar, &QScrollBar::valueChanged, temp, &PDataViewerWidget::updateDataTableTo);
							connect(temp, &PDataViewerWidget::wheelDeltaAngleChange, verticalScrollBar, &QDataViewScrollBar::updateScrollValue);
							temp->setTableScrollBar(verticalScrollBar);

						}


						connect(PSimulationThread::instance(), &PSimulationThread::oneFrameFinished, this, &PScrollBarViewerWidget::updateInstanceField);
						for (auto wid : viewers)
						{
							connect(this, &PScrollBarViewerWidget::updateInstanceWidgets, wid, &PDataViewerWidget::updateDataTable);
						}
					}

					//TriangleSet
					if (field->getTemplateName() == std::string(typeid(TriangleSet<DataType3f>).name())) 
					{

						FArray<TopologyModule::Edge, DeviceType::GPU> edge;
						std::cout<<std::string(typeid(TopologyModule::Edge).name())<<"\n";
						std::cout << std::string(typeid(TopologyModule::Triangle).name()) << "\n";
						std::cout << std::string(typeid(TopologyModule::Edg2Tri).name()) << "\n";
						std::cout << std::string(typeid(TopologyModule::Tri2Edg).name()) << "\n";

						std::vector<PDataViewerWidget*> viewers;
						for (auto it : Name_Field)
						{
							auto f = it.second;
							PDataViewerWidget* temp;
							if (f->mType == std::string(typeid(Vec3f).name()))
							{
								temp = new PVec3FieldViewerWidget(f->mPtr.get());
								viewers.push_back(temp);
							}

							QDataViewScrollBar* verticalScrollBar = new QDataViewScrollBar(Qt::Orientation::Vertical);
							layout->addWidget(temp);
							layout->addWidget(verticalScrollBar);

							connect(verticalScrollBar, &QScrollBar::valueChanged, temp, &PDataViewerWidget::updateDataTableTo);
							connect(temp, &PDataViewerWidget::wheelDeltaAngleChange, verticalScrollBar, &QDataViewScrollBar::updateScrollValue);
							temp->setTableScrollBar(verticalScrollBar);
						}


						connect(PSimulationThread::instance(), &PSimulationThread::oneFrameFinished, this, &PScrollBarViewerWidget::updateInstanceField);
						for (auto wid : viewers)
						{
							connect(this, &PScrollBarViewerWidget::updateInstanceWidgets, wid, &PDataViewerWidget::updateDataTable);
						}


					}
					//EdgeSet
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
			updateInstanceData(mfield);
			emit updateInstanceWidgets();
		}
	

	protected:

		void resizeEvent(QResizeEvent* event)override
		{
			QWidget::resizeEvent(event);
		}



		void closeEvent(QCloseEvent* event) override
		{
			for (auto it : Name_Field)
			{
				auto f = it.second;
				PDataViewerWidget* temp;
				if (f->mType == std::string(typeid(Vec3f).name()))
				{
					temp = new PVec3FieldViewerWidget(f->mPtr.get());
					temp->clear();
					delete temp;
				}
			}
			event->accept(); 
		}



	private:

		FBase* mfield = NULL;

		std::map<std::string, std::shared_ptr<Field_Type>> Name_Field;

		void updateInstanceData(FBase* field)
		{
			// PointSet
			{
				FInstance<PointSet<DataType3f>>* ptSet = TypeInfo::cast<FInstance<PointSet<DataType3f>>>(field);

				//mCoord
				auto points = ptSet->getDataPtr()->getPoints();
				static FArray<Vec3f, DeviceType::GPU> f_points;
				f_points.assign(points);

				static std::shared_ptr<FBase> basePointsPtr = std::static_pointer_cast<FBase>(std::make_shared<FArray<Vec3f, DeviceType::GPU>>(f_points));
				Name_Field[std::string("mCoord")] = std::make_shared<Field_Type>(Field_Type(basePointsPtr, std::string(typeid(Vec3f).name())));
			}
			// PointSet
			{
			}

		
		}

	};

	
	
}
