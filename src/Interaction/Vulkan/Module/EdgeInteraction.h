#pragma once
#include "Module/MouseInputModule.h"
#include "Module/TopologyModule.h"
#include "Topology/TriangleSet.h"
#include "Catalyzer/VkMin.h"

namespace dyno
{
	template<typename TDataType>
	class EdgeInteraction : public MouseInputModule
	{
		DECLARE_TCLASS(EdgeInteraction, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Edge Edge;
		typedef typename TopologyModule::Triangle Triangle;

		EdgeInteraction();
		virtual ~EdgeInteraction() {};

		void calcIntersectClick();
		void calcIntersectDrag();
		void calcEdgeIntersectClick();
		void calcEdgeIntersectDrag();

		void mergeIndex();

		void printInfoClick();
		void printInfoDragging();
		void printInfoDragRelease();

		DEF_INSTANCE_IN(EdgeSet3f, InitialEdgeSet, "");
		DEF_INSTANCE_OUT(EdgeSet3f, SelectedEdgeSet, "");
		DEF_INSTANCE_OUT(EdgeSet3f, OtherEdgeSet, "");
		DEF_ARRAY_OUT(int, EdgeIndex, DeviceType::GPU, "");

		DECLARE_ENUM(PickingTypeSelection,
		Click = 0,
			Drag = 1,
			Both = 2
			);

		DEF_ENUM(PickingTypeSelection, EdgePickingType, PickingTypeSelection::Both, "");

		DECLARE_ENUM(MultiSelectionType,
		OR = 0,
			XOR = 1,
			C = 2
			);

		DEF_ENUM(MultiSelectionType, MultiSelectionType, MultiSelectionType::OR, "");

		DEF_VAR(Real, InteractionRadius, 0.01, "The radius of interaction");

		DEF_VAR(bool, TogglePicker, true, "The toggle of picker");

		DEF_VAR(bool, ToggleMultiSelect, false, "The toggle of multiple selection");

		DEF_VAR(bool, ToggleIndexOutput, true, "The toggle of index output");

	protected:
		void onEvent(PMouseEvent event) override;
	private:
		std::shared_ptr<Camera> camera;
		TRay3D<Real> ray1, ray2;
		Real x1, y1, x2, y2;
		bool isPressed;
		int tempNumT, tempNumS;

		DArray<int> edgeIntersectedIndex;
		DArray<int> tempEdgeIntersectedIndex;

        VkMinElement<Real> mMin;
        VkReduce<int> mReduce;
        VkScan<int> mScan;
	};

	IMPLEMENT_TCLASS(EdgeInteraction, TDataType)
}
