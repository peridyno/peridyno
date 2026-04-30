#pragma once
#include "PInstanceViewerWidget.h"
#include "Topology/TriangleSet.h"
#include "Topology/DiscreteElements.h"
#include "Mapping/DiscreteElementsToTriangleSet.h"


namespace dyno
{
	class PDiscreteElementViewerWidget : public PInstanceViewerWidget
	{
		Q_OBJECT

	public:

		PDiscreteElementViewerWidget(FBase* field, QWidget* pParent = NULL);

	public slots:

		void updateRenderWidget();

	protected:
		void closeEvent(QCloseEvent* event);

		class GLMeshRenderWidget* renderWidget = NULL;
		FInstance<DiscreteElements<DataType3f>>* f_discreteElement = NULL;
	};

}