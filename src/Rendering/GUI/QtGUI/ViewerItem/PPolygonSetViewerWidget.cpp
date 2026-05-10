#include "PPolygonSetViewerWidget.h"
#include <QHeaderView>
#include <QLabel>
#include <QVBoxLayout>
#include <QTabWidget>
#include "Framework/FInstance.h"
#include "Topology/TriangleSet.h"
#include "PVec3FieldViewerWidget.h"
#include "PVec2FieldViewerWidget.h"
#include "PSimulationThread.h"
#include "PIntegerViewerWidget.h"

namespace dyno
{

	PPolygonSetViewerWidget::PPolygonSetViewerWidget(FBase* field, QWidget* pParent) :
	PInstanceViewerWidget(field, pParent)
{
	mfield = field;

	auto layout = new QVBoxLayout();
	this->setLayout(layout);

	f_polySet = TypeInfo::cast<FInstance<PolygonSet<DataType3f>>>(field);

	auto ptCount = new QLabel("Points:");
	auto triCount = new QLabel("Polygons:");

	if (f_polySet)
	{
		ptCount->setText((std::string("Points: ") + std::to_string(f_polySet->constDataPtr()->getPoints().size())).c_str());
		triCount->setText((std::string("Polygons: ") + std::to_string(f_polySet->constDataPtr()->polygonIndices().size())).c_str());
	}


	layout->addWidget(ptCount);
	layout->addWidget(triCount);

	// Create tab widget with horizontal tabs
	QTabWidget* tabWidget = new QTabWidget(this);
	tabWidget->setTabPosition(QTabWidget::North);

	// Create three tabs
	QWidget* tab1 = new QWidget();
	QWidget* tab3 = new QWidget();

	// Add some content to each tab for demonstration
	QLabel* label1 = new QLabel("Content for Tab 1", tab1);
	QLabel* label3 = new QLabel("Content for Tab 3", tab3);

	QHBoxLayout* tab1Layout = new QHBoxLayout(tab1);
	tab1Layout->addWidget(label1);
	tab1->setLayout(tab1Layout);

	QHBoxLayout* tab3Layout = new QHBoxLayout(tab3);
	tab3Layout->addWidget(label3);
	tab3->setLayout(tab3Layout);

	// Add tabs to the tab widget
	tabWidget->addTab(tab1, "Points");
	tabWidget->addTab(tab3, "Polygons");

	// Add the tab widget to the main layout
	layout->addWidget(tabWidget);

	if (f_polySet)
	{
		//points
		{
			f_points = std::make_shared<FArray<Vec3f, DeviceType::GPU>>();
			auto points = f_polySet->constDataPtr()->getPoints();
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

		//triangles
		{
			f_polygons = std::make_shared<FArrayList<uint, DeviceType::GPU>>();
			auto polygons = f_polySet->constDataPtr()->polygonIndices();
			f_polygons->assign(polygons);

			polygonViewer = new PIntegerViewerWidget(f_polygons.get());

			QDataViewScrollBar* verticalScrollBar = new QDataViewScrollBar(Qt::Orientation::Vertical);
			layout->addWidget(polygonViewer);
			layout->addWidget(verticalScrollBar);

			connect(verticalScrollBar, &QScrollBar::valueChanged, polygonViewer, &PDataViewerWidget::updateDataTableTo);
			connect(polygonViewer, &PDataViewerWidget::wheelDeltaAngleChange, verticalScrollBar, &QDataViewScrollBar::updateScrollValue);
			connect(PSimulationThread::instance(), &PSimulationThread::oneFrameFinished, this, &PInstanceViewerWidget::updateWidget);

			polygonViewer->setTableScrollBar(verticalScrollBar);
			verticalScrollBar->resize(verticalScrollBar->width(), this->height());
			tab3Layout->addWidget(polygonViewer, 1);
			tab3Layout->addWidget(verticalScrollBar);
		}
		
	}

	connect(PSimulationThread::instance(), &PSimulationThread::sceneGraphChanged, this, &QWidget::close);


	}
	
	void PPolygonSetViewerWidget::updateWidget()
	{
		if (f_polySet)
		{
			auto points = f_polySet->constDataPtr()->getPoints();
			f_points->assign(points);
			auto polygons = f_polySet->constDataPtr()->polygonIndices();
			f_polygons->assign(polygons);
		}
		else
			return;

		if(pointViewer)
			pointViewer->updateDataTable();

		if (polygonViewer)
			polygonViewer->updateDataTable();

	}



}