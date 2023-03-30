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
	class PointPickerNode : public Node
	{
		DECLARE_TCLASS(PointPickerNode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DEF_INSTANCE_IN(PointSet<TDataType>, Topology, "");

		DEF_ARRAY_STATE(int, PointIndex, DeviceType::GPU, "");

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

		DEF_VAR(Real, InteractionRadius, 0.002f, "The radius of interaction");
		DEF_VAR(Real, PointSelectedSize, 0.012f, "");
		DEF_VAR(Real, PointOtherSize, 0.01f, "");

		PointPickerNode(std::string name = "default");
		~PointPickerNode();

		std::string getNodeType();

		void resetStates() override;

		void changePickingType();
		void changeMultiSelectionType();

	private:
		std::shared_ptr<PointInteraction<TDataType>> pointInteractor;
	};
}