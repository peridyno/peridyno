#pragma once
#include "Node.h"
#include "Topology/QuadSet.h"
#include "Module/TopologyModule.h"
#include "Module/SurfaceInteraction.h"
#include "Module/EdgeInteraction.h"
#include "Module/PointInteraction.h"
#include "Mapping/QuadSetToTriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class QuadPickerNode : public Node
	{
		DECLARE_TCLASS(QuadPickerNode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DEF_INSTANCE_IN(QuadSet<TDataType>, Topology, "");

		DEF_ARRAY_STATE(int, QuadIndex, DeviceType::GPU, "");
		DEF_ARRAY_STATE(int, EdgeIndex, DeviceType::GPU, "");
		DEF_ARRAY_STATE(int, PointIndex, DeviceType::GPU, "");
		DEF_ARRAY_STATE(int, Sur2PointIndex, DeviceType::GPU, "");

		DECLARE_ENUM(PickingElementTypeSelection,
		Surface = 0,
			Edge = 1,
			Point = 2,
			All = 3
			);

		DEF_ENUM(PickingElementTypeSelection, PickingElementType, PickingElementTypeSelection::All, "");

		DECLARE_ENUM(PickingTypeSelection,
		Click = 0,
			Drag = 1,
			Both = 2
			);

		DEF_ENUM(PickingTypeSelection, PickingType, PickingTypeSelection::Both, "");

		DECLARE_ENUM(MultiSelectionType,
		OR = 0,
			XOR = 1,
			C = 2
			);

		DEF_ENUM(MultiSelectionType, MultiSelectionType, MultiSelectionType::OR, "");

		DEF_VAR(bool, ToggleFlood, false, "The toggle of surface flood selection");

		DEF_VAR(bool, ToggleVisibleFilter, true, "The toggle of visible filter");

		DEF_VAR(bool, ToggleIndexOutput, true, "The toggle of index output");

		DEF_VAR(Real, FloodAngle, 1.0f, "The angle limit of surface flood selection");

		DEF_VAR(Real, InteractionRadius, 0.002f, "The radius of interaction");

		DEF_VAR(Real, PointSelectedSize, 0.006f, "");
		DEF_VAR(Real, PointOtherSize, 0.005f, "");

		DEF_VAR(Real, EdgeSelectedSize, 0.002f, "");
		DEF_VAR(Real, EdgeOtherSize, 0.0015, "");

		QuadPickerNode();
		~QuadPickerNode() override;

		std::string getNodeType();

		void resetStates() override;

		void changePickingElementType();
		void changePickingType();
		void changeMultiSelectionType();

	private:
		std::shared_ptr<SurfaceInteraction<TDataType>> surfaceInteractor;
		std::shared_ptr<EdgeInteraction<TDataType>> edgeInteractor;
		std::shared_ptr<PointInteraction<TDataType>> pointInteractor;
		std::shared_ptr<QuadSetToTriangleSet<TDataType>> mapper;
	};
}