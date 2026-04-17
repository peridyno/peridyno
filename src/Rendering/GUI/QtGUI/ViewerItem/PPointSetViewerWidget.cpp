#include "PPointSetViewerWidget.h"
#include <QHeaderView>
#include <QLabel>
#include <QVBoxLayout>
#include <QTabWidget>
#include "Framework/FInstance.h"
#include "Topology/TriangleSet.h"
#include "PVec3FieldViewerWidget.h"
#include "PVec2FieldViewerWidget.h"
#include "PSimulationThread.h"


namespace dyno
{

	PPointSetViewerWidget::PPointSetViewerWidget(FBase* field, QWidget* pParent) :
		PInstanceViewerWidget(field, pParent)
	{
		mfield = field;

		auto layout = new QVBoxLayout();
		this->setLayout(layout);

		f_ptSet = TypeInfo::cast<FInstance<PointSet<DataType3f>>>(field);

		auto ptCount = new QLabel("Points:");

		if (f_ptSet)
			ptCount->setText((std::string("Points: ") + std::to_string(f_ptSet->constDataPtr()->getPoints().size())).c_str());

		layout->addWidget(ptCount);


		// Create tab widget with horizontal tabs
		QTabWidget* tabWidget = new QTabWidget(this);
		tabWidget->setTabPosition(QTabWidget::North);

		// Create three tabs
		QWidget* tab1 = new QWidget();

		QHBoxLayout* tab1Layout = new QHBoxLayout(tab1);
		tab1->setLayout(tab1Layout);

		// Add tabs to the tab widget
		tabWidget->addTab(tab1, "Points");

		// Add the tab widget to the main layout
		layout->addWidget(tabWidget);

		if (f_ptSet)
		{
			//points
			{
				f_points = std::make_shared<FArray<Vec3f, DeviceType::GPU>>();
				auto points = f_ptSet->constDataPtr()->getPoints();
				f_points->assign(points);

				pointViewer = new PVec3FieldViewerWidget(f_points.get());

				QDataViewScrollBar* verticalScrollBar = new QDataViewScrollBar(Qt::Orientation::Vertical);
				layout->addWidget(pointViewer);
				layout->addWidget(verticalScrollBar);

				connect(verticalScrollBar, &QScrollBar::valueChanged, pointViewer, &PDataViewerWidget::updateDataTableTo);
				connect(pointViewer, &PDataViewerWidget::wheelDeltaAngleChange, verticalScrollBar, &QDataViewScrollBar::updateScrollValue);
				connect(PSimulationThread::instance(), &PSimulationThread::oneFrameFinished, this, &PInstanceViewerWidget::updateWidget);

				pointViewer->setTableScrollBar(verticalScrollBar);
				verticalScrollBar->resize(verticalScrollBar->width(), this->height());
				tab1Layout->addWidget(pointViewer, 1);
				tab1Layout->addWidget(verticalScrollBar);
			}

		}

		connect(PSimulationThread::instance(), &PSimulationThread::sceneGraphChanged, this, &QWidget::close);


	}

	void PPointSetViewerWidget::updateWidget()
	{
		if (f_ptSet)
		{
			auto points = f_ptSet->constDataPtr()->getPoints();
			f_points->assign(points);
		}
		else
			return;

		if (pointViewer)
			pointViewer->updateDataTable();

	}



}