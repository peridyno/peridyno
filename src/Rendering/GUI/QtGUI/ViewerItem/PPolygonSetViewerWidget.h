#pragma once
#include "PInstanceViewerWidget.h"
#include "Topology/TriangleSet.h"
#include "Topology/PolygonSet.h"

namespace dyno
{
	class PPolygonSetViewerWidget : public PInstanceViewerWidget
	{
		Q_OBJECT

	public:

		PPolygonSetViewerWidget(FBase* field,QWidget* pParent = NULL);

	public slots:

		void updateWidget()override;


	protected:

		std::shared_ptr<FArray<Vec3f, DeviceType::GPU>> f_points;
		std::shared_ptr<FArrayList<uint, DeviceType::GPU>> f_polygons;

		PDataViewerWidget* pointViewer = NULL;
		PDataViewerWidget* polygonViewer = NULL;

		FInstance<PolygonSet<DataType3f>>* f_polySet = NULL;

	};



}