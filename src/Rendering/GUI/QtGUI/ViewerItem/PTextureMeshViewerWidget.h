#pragma once
#include "PInstanceViewerWidget.h"
#include "Topology/TriangleSet.h"
#include "Topology/TextureMesh.h"

namespace dyno
{
	class PTextureMeshViewerWidget : public PInstanceViewerWidget
	{
		Q_OBJECT

	public:

		PTextureMeshViewerWidget(FBase* field, QWidget* pParent = NULL);

	public slots:

		void updateWidget()override;


	protected:

		std::shared_ptr<FArray<Vec3f, DeviceType::GPU>> f_points;
		std::shared_ptr<FArray<Topology::Edge, DeviceType::GPU>> f_edges;
		std::shared_ptr<FArray<Topology::Triangle, DeviceType::GPU>> f_triangles;

		PDataViewerWidget* pointViewer = NULL;
		PDataViewerWidget* edgeViewer = NULL;
		PDataViewerWidget* triangleViewer = NULL;

		FInstance<TextureMesh>* f_textureMesh = NULL;

	};

}