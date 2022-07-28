#pragma once
#include "Module/MouseInputModule.h"
#include "Module/TopologyModule.h"
#include "Topology/TriangleSet.h"

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

		DEF_INSTANCE_IN(TriangleSet<TDataType>, InitialTriangleSet, "");
		DEF_INSTANCE_OUT(EdgeSet<TDataType>, SelectedEdgeSet, "");
		DEF_INSTANCE_OUT(EdgeSet<TDataType>, OtherEdgeSet, "");
		DEF_ARRAY_OUT(int, EdgeIndex, DeviceType::GPU, "");

		DEF_VAR(Real, InterationRadius, 0.01, "The radius of interaction");

		DEF_VAR(bool, TogglePicker, true, "The toggle of picker");

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

		DArray<int> edgeIntersectedIndex;

		DArray<int> tempEdgeIntersectedIndex;
	};

	IMPLEMENT_TCLASS(EdgeInteraction, TDataType)
}
