#pragma once
#include "Module/MouseInputModule.h"
#include "Module/TopologyModule.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class PickerInteraction : public MouseInputModule
	{
		DECLARE_TCLASS(PickerInteraction, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Edge Edge;
		typedef typename TopologyModule::Triangle Triangle;

		PickerInteraction();
		virtual ~PickerInteraction() {};

		void calcIntersectClick();
		void calcIntersectDrag();
		void calcSurfaceIntersectClick();
		void calcSurfaceIntersectDrag();
		void calcEdgeIntersectClick();
		void calcEdgeIntersectDrag();
		void calcPointIntersectClick();
		void calcPointIntersectDrag();

		DEF_INSTANCE_IN(TriangleSet<TDataType>, InitialTriangleSet, "");
		DEF_INSTANCE_OUT(TriangleSet<TDataType>, SelectedTriangleSet, "");
		DEF_INSTANCE_OUT(TriangleSet<TDataType>, OtherTriangleSet, "");
		DEF_ARRAY_OUT(int, TriangleIndex, DeviceType::GPU, "");
		DEF_INSTANCE_OUT(EdgeSet<TDataType>, SelectedEdgeSet, "");
		DEF_INSTANCE_OUT(EdgeSet<TDataType>, OtherEdgeSet, "");
		DEF_ARRAY_OUT(int, EdgeIndex, DeviceType::GPU, "");
		DEF_INSTANCE_OUT(PointSet<TDataType>, SelectedPointSet, "");
		DEF_INSTANCE_OUT(PointSet<TDataType>, OtherPointSet, "");
		DEF_ARRAY_OUT(int, PointIndex, DeviceType::GPU, "");

		DEF_VAR(Real, InterationRadius, 0.01, "The radius of interaction");
		DEF_VAR(bool, ToggleSurfacePicker, true, "The toggle of surface picker");
		DEF_VAR(bool, ToggleEdgePicker, true, "The toggle of edge picker");
		DEF_VAR(bool, TogglePointPicker,true, "The toggle of point picker");

		DEF_VAR(bool, ToggleMultiSelect, false, "The toggle of multiple selection");

	protected:
		void onEvent(PMouseEvent event) override;
	private:
		std::shared_ptr<Camera> camera;
		TRay3D<Real> ray1;
		TRay3D<Real> ray2;
		Real x1;
		Real y1;
		Real x2;
		Real y2;
		bool isPressed;

		DArray<int> triIntersectedIndex;
		DArray<int> edgeIntersectedIndex;
		DArray<int> pointIntersectedIndex;
	};

	IMPLEMENT_TCLASS(PickerInteraction, TDataType)
}
