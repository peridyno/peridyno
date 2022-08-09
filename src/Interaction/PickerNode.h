#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Module/TopologyModule.h"
#include "SurfaceInteraction.h"
#include "EdgeInteraction.h"
#include "PointInteraction.h"

namespace dyno
{
	template<typename TDataType>
	class PickerNode : public Node
	{
		DECLARE_TCLASS(PickerNode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DEF_INSTANCE_IN(TriangleSet<TDataType>, Topology, "");

		DEF_ARRAY_STATE(int, TriangleIndex, DeviceType::GPU, "");
		DEF_ARRAY_STATE(int, EdgeIndex, DeviceType::GPU, "");
		DEF_ARRAY_STATE(int, PointIndex, DeviceType::GPU, "");

		DEF_VAR(Real, InterationRadius, 0.01f, "The radius of interaction");
		DEF_VAR(bool, ToggleSurfacePicker, true, "The toggle for surface picker");
		DEF_VAR(bool, ToggleEdgePicker, true, "The toggle for edge picker");
		DEF_VAR(bool, TogglePointPicker, true, "The toggle for point picker");

		DECLARE_ENUM(PickingTypeSelection,
		Click = 0,
			Drag = 1,
			Both = 2
			);

		DEF_ENUM(PickingTypeSelection, PickingType, PickingTypeSelection::Both, "");

		DECLARE_ENUM(PickingElementTypeSelection,
		Surface = 0,
			Edge = 1,
			Point = 2
			);

		DEF_ENUM(PickingElementTypeSelection, PickingElementType, PickingElementTypeSelection::Surface, "");

		DECLARE_ENUM(MultiSelectionType,
		OR = 0,
			XOR = 1,
			C = 2
			);

		DEF_ENUM(MultiSelectionType, MultiSelectionType, MultiSelectionType::OR, "");

		DEF_VAR(Vec3f, SelectedTriangleColor, Vec3f(0.2, 0.48, 0.75), "");
		DEF_VAR(Vec3f, OtherTriangleColor, Vec3f(0.8, 0.52, 0.25), "");
		DEF_VAR(Vec3f, SelectedEdgeColor, Vec3f(0.8f, 0.0f, 0.0f), "");
		DEF_VAR(Vec3f, OtherEdgeColor, Vec3f(0.0f), "");
		DEF_VAR(Vec3f, SelectedPointColor, Vec3f(1.0f, 0, 0), "");
		DEF_VAR(Vec3f, OtherPointColor, Vec3f(0, 0, 1.0f), "");

		DEF_VAR(Real, SelectedSize, 0.02f, "");
		DEF_VAR(Real, OtherSize, 0.01f, "");

		PickerNode(std::string name = "default");
		~PickerNode();

		std::string getNodeType();

		void resetStates() override;

		void changePickingElementType();

	private:
		std::shared_ptr<SurfaceInteraction<TDataType>> surfaceInteractor;
		std::shared_ptr<EdgeInteraction<TDataType>> edgeInteractor;
		std::shared_ptr<PointInteraction<TDataType>> pointInteractor;
	};
}