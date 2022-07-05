#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Module/TopologyModule.h"
#include "PickerInteraction.h"

namespace dyno
{
	template<typename TDataType>
	class PickerNode : public Node
	{
		DECLARE_TCLASS(PickerNode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, InTopology, "");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, SelectedTriangleSet, "");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, OtherTriangleSet, "");
		DEF_INSTANCE_STATE(EdgeSet<TDataType>, SelectedEdgeSet, "");
		DEF_INSTANCE_STATE(EdgeSet<TDataType>, OtherEdgeSet, "");
		DEF_INSTANCE_STATE(PointSet<TDataType>, SelectedPointSet, "");
		DEF_INSTANCE_STATE(PointSet<TDataType>, OtherPointSet, "");

		DEF_VAR(Real, InterationRadius, 0.05f, "The radius of interaction");
		DEF_VAR(bool, ToggleSurfacePicker, true, "The toggle for surface picker");
		DEF_VAR(bool, ToggleEdgePicker, true, "The toggle for edge picker");
		DEF_VAR(bool, TogglePointPicker, true, "The toggle for point picker");

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

	private:
		std::shared_ptr<PickerInteraction<TDataType>> mouseInteractor;
	};
}