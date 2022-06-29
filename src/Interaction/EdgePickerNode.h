#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Module/TopologyModule.h"
#include "EdgeInteraction.h"
namespace dyno
{
	template<typename TDataType>
	class EdgePickerNode : public Node
	{
		DECLARE_TCLASS(EdgePickerNode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DEF_INSTANCE_IN(TriangleSet<TDataType>, InTopology, "");
		DEF_INSTANCE_STATE(EdgeSet<TDataType>, SelectedTopology, "");
		DEF_INSTANCE_STATE(EdgeSet<TDataType>, OtherTopology, "");
		DEF_INSTANCE_STATE(EdgeIteraction<TDataType>, MouseInteractor, "");

		EdgePickerNode(std::string name = "default");
		~EdgePickerNode();

		void resetStates() override;
	};
}