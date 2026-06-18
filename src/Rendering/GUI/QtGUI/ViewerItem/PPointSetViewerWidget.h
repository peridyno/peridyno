#pragma once
#include "PInstanceViewerWidget.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	class PPointSetViewerWidget : public PInstanceViewerWidget
	{
		Q_OBJECT

	public:

		PPointSetViewerWidget(FBase* field, QWidget* pParent = NULL);

	public slots:

		void updateWidget()override;


	protected:

		std::shared_ptr<FArray<Vec3f, DeviceType::GPU>> f_points;

		PDataViewerWidget* pointViewer = NULL;

		FInstance<PointSet<DataType3f>>* f_ptSet = NULL;


	};



}