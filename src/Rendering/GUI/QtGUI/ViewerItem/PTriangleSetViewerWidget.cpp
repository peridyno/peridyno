#include "PTriangleSetViewerWidget.h"
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

	PTriangleSetViewerWidget::PTriangleSetViewerWidget(FBase* field, QWidget* pParent) :
	PInstanceViewerWidget(field, pParent)
{
	mfield = field;

	auto layout = new QVBoxLayout();
	this->setLayout(layout);

	f_triSet = TypeInfo::cast<FInstance<TriangleSet<DataType3f>>>(field);

	auto ptCount = new QLabel("Points:");
	auto edgeCount = new QLabel("Edges:");
	auto triCount = new QLabel("Triangles:");

	if (f_triSet)
	{
		ptCount->setText((std::string("Points: ") + std::to_string(f_triSet->constDataPtr()->getPoints().size())).c_str());
		edgeCount->setText((std::string("Edges: ") + std::to_string(f_triSet->constDataPtr()->edgeIndices().size())).c_str());
		triCount->setText((std::string("Triangles: ") + std::to_string(f_triSet->constDataPtr()->triangleIndices().size())).c_str());
	}


	layout->addWidget(ptCount);
	layout->addWidget(edgeCount);
	layout->addWidget(triCount);

	// Create tab widget with horizontal tabs
	QTabWidget* tabWidget = new QTabWidget(this);
	tabWidget->setTabPosition(QTabWidget::North);

	// Create three tabs
	QWidget* tab1 = new QWidget();
	QWidget* tab2 = new QWidget();
	QWidget* tab3 = new QWidget();

	// Add some content to each tab for demonstration
	QLabel* label1 = new QLabel("Content for Tab 1", tab1);
	QLabel* label2 = new QLabel("Content for Tab 2", tab2);
	QLabel* label3 = new QLabel("Content for Tab 3", tab3);

	QHBoxLayout* tab1Layout = new QHBoxLayout(tab1);
	tab1Layout->addWidget(label1);
	tab1->setLayout(tab1Layout);

	QHBoxLayout* tab2Layout = new QHBoxLayout(tab2);
	tab2Layout->addWidget(label2);
	tab2->setLayout(tab2Layout);

	QHBoxLayout* tab3Layout = new QHBoxLayout(tab3);
	tab3Layout->addWidget(label3);
	tab3->setLayout(tab3Layout);

	// Add tabs to the tab widget
	tabWidget->addTab(tab1, "Points");
	tabWidget->addTab(tab2, "Edges");
	tabWidget->addTab(tab3, "Triangles");

	// Add the tab widget to the main layout
	layout->addWidget(tabWidget);

	if (f_triSet)
	{
		//points
		{
			f_points = std::make_shared<FArray<Vec3f, DeviceType::GPU>>();
			auto points = f_triSet->constDataPtr()->getPoints();
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
			f_triangles = std::make_shared<FArray<Topology::Triangle, DeviceType::GPU>>();
			auto triangles = f_triSet->constDataPtr()->triangleIndices();
			f_triangles->assign(triangles);

			triangleViewer = new PVec3FieldViewerWidget(f_triangles.get());

			QDataViewScrollBar* verticalScrollBar = new QDataViewScrollBar(Qt::Orientation::Vertical);
			layout->addWidget(triangleViewer);
			layout->addWidget(verticalScrollBar);

			connect(verticalScrollBar, &QScrollBar::valueChanged, triangleViewer, &PDataViewerWidget::updateDataTableTo);
			connect(triangleViewer, &PDataViewerWidget::wheelDeltaAngleChange, verticalScrollBar, &QDataViewScrollBar::updateScrollValue);
			connect(PSimulationThread::instance(), &PSimulationThread::oneFrameFinished, this, &PInstanceViewerWidget::updateWidget);

			triangleViewer->setTableScrollBar(verticalScrollBar);
			verticalScrollBar->resize(verticalScrollBar->width(), this->height());
			tab3Layout->addWidget(triangleViewer, 1);
			tab3Layout->addWidget(verticalScrollBar);
		}
		//edges
		{
			f_edges = std::make_shared<FArray<Topology::Edge, DeviceType::GPU>>();
			auto edges = f_triSet->constDataPtr()->edgeIndices();
			f_edges->assign(edges);

			edgeViewer = new PVec2FieldViewerWidget(f_edges.get());

			QDataViewScrollBar* verticalScrollBar = new QDataViewScrollBar(Qt::Orientation::Vertical);
			layout->addWidget(edgeViewer);
			layout->addWidget(verticalScrollBar);

			connect(verticalScrollBar, &QScrollBar::valueChanged, edgeViewer, &PDataViewerWidget::updateDataTableTo);
			connect(edgeViewer, &PDataViewerWidget::wheelDeltaAngleChange, verticalScrollBar, &QDataViewScrollBar::updateScrollValue);
			connect(PSimulationThread::instance(), &PSimulationThread::oneFrameFinished, this, &PInstanceViewerWidget::updateWidget);

			edgeViewer->setTableScrollBar(verticalScrollBar);
			verticalScrollBar->resize(verticalScrollBar->width(), this->height());
			tab2Layout->addWidget(edgeViewer, 1);
			tab2Layout->addWidget(verticalScrollBar);
		}
	}

	connect(PSimulationThread::instance(), &PSimulationThread::sceneGraphChanged, this, &QWidget::close);


	}
	
	void PTriangleSetViewerWidget::updateWidget() 
	{
		if (f_triSet)
		{
			auto points = f_triSet->constDataPtr()->getPoints();
			f_points->assign(points);
			auto edges = f_triSet->constDataPtr()->edgeIndices();
			f_edges->assign(edges);
			auto triangles = f_triSet->constDataPtr()->triangleIndices();
			f_triangles->assign(triangles);
		}
		else
			return;

		if(pointViewer)
			pointViewer->updateDataTable();

		if (edgeViewer)
			edgeViewer->updateDataTable();

		if (triangleViewer)
			triangleViewer->updateDataTable();

	}



}