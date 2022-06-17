#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Module/TopologyModule.h"
#include "CustomMouseInteraction.h"
namespace dyno
{
	template<typename TDataType>
	class SurfacePickerNode : public Node
	{
		DECLARE_TCLASS(SurfacePickerNode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DEF_INSTANCE_STATE(TriangleSet<TDataType>,InTopology,"");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, SelectedTopology, "");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, OtherTopology, "");
		DEF_INSTANCE_STATE(CustomMouseIteraction<TDataType>, MouseInteractor, "");

		SurfacePickerNode(std::string name = "default");
		~SurfacePickerNode();

		void resetStates() override;
	};
}