#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Module/TopologyModule.h"
#include "SurfaceInteraction.h"
namespace dyno
{
	template<typename TDataType>
	class SurfacePickerNode : public Node
	{
		DECLARE_TCLASS(SurfacePickerNode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		DEF_INSTANCE_IN(TriangleSet<TDataType>,InTopology,"");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, SelectedTopology, "");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, OtherTopology, "");

		SurfacePickerNode(std::string name = "default");
		~SurfacePickerNode();

		std::string getNodeType();

		void resetStates() override;

	private:
		std::shared_ptr<SurfaceInteraction<TDataType>> mouseInteractor;
	};
}