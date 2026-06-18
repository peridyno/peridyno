#pragma once
#include "PInstanceViewerWidget.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	class PTriangleSetViewerWidget : public PInstanceViewerWidget
	{
		Q_OBJECT

	public:

		PTriangleSetViewerWidget(FBase* field,QWidget* pParent = NULL);

	public slots:

		void updateWidget()override;


	protected:

		std::shared_ptr<FArray<Vec3f, DeviceType::GPU>> f_points;
		std::shared_ptr<FArray<Topology::Edge, DeviceType::GPU>> f_edges;
		std::shared_ptr<FArray<Topology::Triangle, DeviceType::GPU>> f_triangles;

		PDataViewerWidget* pointViewer = NULL;
		PDataViewerWidget* edgeViewer = NULL;
		PDataViewerWidget* triangleViewer = NULL;

		FInstance<TriangleSet<DataType3f>>* f_triSet = NULL;

	};



}