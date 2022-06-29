#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Module/TopologyModule.h"
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

		DEF_INSTANCE_IN(TriangleSet<TDataType>, InTopology, "");
		DEF_INSTANCE_STATE(PointSet<TDataType>, SelectedTopology, "");
		DEF_INSTANCE_STATE(PointSet<TDataType>, OtherTopology, "");
		DEF_INSTANCE_STATE(PointIteraction<TDataType>, MouseInteractor, "");

		PointPickerNode(std::string name = "default");
		~PointPickerNode();

		void resetStates() override;
	};
}