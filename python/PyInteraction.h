#pragma once
#include "PyCommon.h"

#include "Module/EdgeInteraction.h"
template <typename TDataType>
void declare_edge_interaction(py::module& m, std::string typestr) {
	using Class = dyno::EdgeInteraction<TDataType>;
	using Parent = dyno::MouseInputModule;

	class EdgeInteractionTrampoline : public Class
	{
	public:
		using Class::Class;

		void onEvent(dyno::PMouseEvent event) override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::EdgeInteraction<TDataType>,
				onEvent,
				event
			);
		}
	};

	class EdgeInteractionPublicist : public Class
	{
	public:
		using Class::onEvent;
	};

	std::string pyclass_name = std::string("EdgeInteraction") + typestr;
	py::class_<Class, Parent, EdgeInteractionTrampoline, std::shared_ptr<Class>>EI(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	EI.def(py::init<>())
		.def("calcIntersectClick", &Class::calcIntersectClick)
		.def("calcIntersectDrag", &Class::calcIntersectDrag)
		.def("calcEdgeIntersectClick", &Class::calcEdgeIntersectClick)
		.def("calcEdgeIntersectDrag", &Class::calcEdgeIntersectDrag)

		.def("mergeIndex", &Class::mergeIndex)

		.def("printInfoClick", &Class::printInfoClick)
		.def("printInfoDragging", &Class::printInfoDragging)
		.def("printInfoDragRelease", &Class::printInfoDragRelease)

		.def("inInitialEdgeSet", &Class::inInitialEdgeSet, py::return_value_policy::reference)
		.def("outSelectedEdgeSet", &Class::outSelectedEdgeSet, py::return_value_policy::reference)
		.def("outOtherEdgeSet", &Class::outOtherEdgeSet, py::return_value_policy::reference)
		.def("outEdgeIndex", &Class::outEdgeIndex, py::return_value_policy::reference)

		.def("varEdgePickingType", &Class::varEdgePickingType, py::return_value_policy::reference)

		.def("varMultiSelectionType", &Class::varMultiSelectionType, py::return_value_policy::reference)

		.def("varInteractionRadius", &Class::varInteractionRadius, py::return_value_policy::reference)
		.def("varTogglePicker", &Class::varTogglePicker, py::return_value_policy::reference)

		.def("varToggleMultiSelect", &Class::varToggleMultiSelect, py::return_value_policy::reference)
		.def("varToggleIndexOutput", &Class::varToggleIndexOutput, py::return_value_policy::reference)
		// protected
		.def("onEvent", &EdgeInteractionPublicist::onEvent);

	py::enum_<typename Class::PickingTypeSelection>(EI, "PickingTypeSelection")
		.value("Click", Class::PickingTypeSelection::Click)
		.value("Drag", Class::PickingTypeSelection::Drag)
		.value("Both", Class::PickingTypeSelection::Both);

	py::enum_<typename Class::MultiSelectionType>(EI, "MultiSelectionType")
		.value("OR", Class::MultiSelectionType::OR)
		.value("XOR", Class::MultiSelectionType::XOR)
		.value("C", Class::MultiSelectionType::C);
}

#include "Module/PointInteraction.h"
template <typename TDataType>
void declare_point_interaction(py::module& m, std::string typestr) {
	using Class = dyno::PointInteraction<TDataType>;
	using Parent = dyno::MouseInputModule;

	class PointInteractionTrampoline : public Class
	{
	public:
		using Class::Class;

		void onEvent(dyno::PMouseEvent event) override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PointInteraction<TDataType>,
				onEvent,
				event
			);
		}
	};

	class PointInteractionPublicist : public Class
	{
	public:
		using Class::onEvent;
	};

	std::string pyclass_name = std::string("PointInteraction") + typestr;
	py::class_<Class, Parent, PointInteractionTrampoline, std::shared_ptr<Class>>PI(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	PI.def(py::init<>())
		.def("calcIntersectClick", &Class::calcIntersectClick)
		.def("calcIntersectDrag", &Class::calcIntersectDrag)
		.def("calcPointIntersectClick", &Class::calcPointIntersectClick)
		.def("calcPointIntersectDrag", &Class::calcPointIntersectDrag)

		.def("mergeIndex", &Class::mergeIndex)

		.def("printInfoClick", &Class::printInfoClick)
		.def("printInfoDragging", &Class::printInfoDragging)
		.def("printInfoDragRelease", &Class::printInfoDragRelease)

		.def("inInitialPointSet", &Class::inInitialPointSet, py::return_value_policy::reference)
		.def("outSelectedPointSet", &Class::outSelectedPointSet, py::return_value_policy::reference)
		.def("outOtherPointSet", &Class::outOtherPointSet, py::return_value_policy::reference)
		.def("outPointIndex", &Class::outPointIndex, py::return_value_policy::reference)

		.def("varPointPickingType", &Class::varPointPickingType, py::return_value_policy::reference)

		.def("varMultiSelectionType", &Class::varMultiSelectionType, py::return_value_policy::reference)

		.def("varInteractionRadius", &Class::varInteractionRadius, py::return_value_policy::reference)
		.def("varTogglePicker", &Class::varTogglePicker, py::return_value_policy::reference)

		.def("varToggleMultiSelect", &Class::varToggleMultiSelect, py::return_value_policy::reference)
		.def("varToggleIndexOutput", &Class::varToggleIndexOutput, py::return_value_policy::reference)
		// protected
		.def("onEvent", &PointInteractionPublicist::onEvent);

	py::enum_<typename Class::PickingTypeSelection>(PI, "PickingTypeSelection")
		.value("Click", Class::PickingTypeSelection::Click)
		.value("Drag", Class::PickingTypeSelection::Drag)
		.value("Both", Class::PickingTypeSelection::Both);

	py::enum_<typename Class::MultiSelectionType>(PI, "MultiSelectionType")
		.value("OR", Class::MultiSelectionType::OR)
		.value("XOR", Class::MultiSelectionType::XOR)
		.value("C", Class::MultiSelectionType::C);
}

#include "Module/SurfaceInteraction.h"
template <typename TDataType>
void declare_surface_interaction(py::module& m, std::string typestr) {
	using Class = dyno::SurfaceInteraction<TDataType>;
	using Parent = dyno::MouseInputModule;

	class SurfaceInteractionTrampoline : public Class
	{
	public:
		using Class::Class;

		void onEvent(dyno::PMouseEvent event) override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SurfaceInteraction<TDataType>,
				onEvent,
				event
			);
		}
	};

	class SurfaceInteractionPublicist : public Class
	{
	public:
		using Class::onEvent;
	};

	std::string pyclass_name = std::string("SurfaceInteraction") + typestr;
	py::class_<Class, Parent, SurfaceInteractionTrampoline, std::shared_ptr<Class>>SI(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	SI.def(py::init<>())
		.def("calcIntersectClick", &Class::calcIntersectClick)
		.def("calcIntersectDrag", &Class::calcIntersectDrag)
		.def("calcSurfaceIntersectClick", &Class::calcSurfaceIntersectClick)
		.def("calcSurfaceIntersectDrag", &Class::calcSurfaceIntersectDrag)

		.def("mergeIndex", &Class::mergeIndex)

		.def("printInfoClick", &Class::printInfoClick)
		.def("printInfoDragging", &Class::printInfoDragging)
		.def("printInfoDragRelease", &Class::printInfoDragRelease)

		.def("inInitialTriangleSet", &Class::inInitialTriangleSet, py::return_value_policy::reference)
		.def("outSelectedTriangleSet", &Class::outSelectedTriangleSet, py::return_value_policy::reference)
		.def("outOtherTriangleSet", &Class::outOtherTriangleSet, py::return_value_policy::reference)
		.def("outTriangleIndex", &Class::outTriangleIndex, py::return_value_policy::reference)
		.def("outSur2PointIndex", &Class::outSur2PointIndex, py::return_value_policy::reference)

		.def("varSurfacePickingType", &Class::varSurfacePickingType, py::return_value_policy::reference)

		.def("varMultiSelectionType", &Class::varMultiSelectionType, py::return_value_policy::reference)

		.def("varFloodAngle", &Class::varFloodAngle, py::return_value_policy::reference)
		.def("varTogglePicker", &Class::varTogglePicker, py::return_value_policy::reference)
		.def("varToggleMultiSelect", &Class::varToggleMultiSelect, py::return_value_policy::reference)

		.def("varToggleFlood", &Class::varToggleFlood, py::return_value_policy::reference)
		.def("varToggleVisibleFilter", &Class::varToggleVisibleFilter, py::return_value_policy::reference)

		.def("varToggleQuad", &Class::varToggleQuad, py::return_value_policy::reference)
		.def("varToggleIndexOutput", &Class::varToggleIndexOutput, py::return_value_policy::reference)
		// protected
		.def("onEvent", &SurfaceInteractionPublicist::onEvent);

	py::enum_<typename Class::PickingTypeSelection>(SI, "PickingTypeSelection")
		.value("Click", Class::PickingTypeSelection::Click)
		.value("Drag", Class::PickingTypeSelection::Drag)
		.value("Both", Class::PickingTypeSelection::Both);

	py::enum_<typename Class::MultiSelectionType>(SI, "MultiSelectionType")
		.value("OR", Class::MultiSelectionType::OR)
		.value("XOR", Class::MultiSelectionType::XOR)
		.value("C", Class::MultiSelectionType::C);
}

#include "EdgePickerNode.h"
template <typename TDataType>
void declare_edge_picker_node(py::module& m, std::string typestr) {
	using Class = dyno::EdgePickerNode<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("EdgePickerNode") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>EPN(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	EPN.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("resetStates", &Class::resetStates)

		.def("changePickingElementType", &Class::changePickingElementType)
		.def("changePickingType", &Class::changePickingType)
		.def("changeMultiSelectionType", &Class::changeMultiSelectionType)

		.def("inTopology", &Class::inTopology, py::return_value_policy::reference)

		.def("stateEdgeIndex", &Class::stateEdgeIndex, py::return_value_policy::reference)
		.def("statePointIndex", &Class::statePointIndex, py::return_value_policy::reference)

		.def("varPickingElementType", &Class::varPickingElementType, py::return_value_policy::reference)

		.def("varPickingType", &Class::varPickingType, py::return_value_policy::reference)

		.def("varMultiSelectionType", &Class::varMultiSelectionType, py::return_value_policy::reference)

		.def("varInteractionRadius", &Class::varInteractionRadius, py::return_value_policy::reference)
		.def("varPointSelectedSize", &Class::varPointSelectedSize, py::return_value_policy::reference)
		.def("varPointOtherSize", &Class::varPointOtherSize, py::return_value_policy::reference)

		.def("varEdgeSelectedSize", &Class::varEdgeSelectedSize, py::return_value_policy::reference)
		.def("varEdgeOtherSize", &Class::varEdgeOtherSize, py::return_value_policy::reference)

		.def("varToggleIndexOutput", &Class::varToggleIndexOutput, py::return_value_policy::reference);

	py::enum_<typename Class::PickingElementTypeSelection>(EPN, "PickingElementTypeSelection")
		.value("Edge", Class::PickingElementTypeSelection::Edge)
		.value("Point", Class::PickingElementTypeSelection::Point)
		.value("All", Class::PickingElementTypeSelection::All);

	py::enum_<typename Class::PickingTypeSelection>(EPN, "PickingTypeSelection")
		.value("Click", Class::PickingTypeSelection::Click)
		.value("Drag", Class::PickingTypeSelection::Drag)
		.value("Both", Class::PickingTypeSelection::Both);

	py::enum_<typename Class::MultiSelectionType>(EPN, "MultiSelectionType")
		.value("OR", Class::MultiSelectionType::OR)
		.value("XOR", Class::MultiSelectionType::XOR)
		.value("C", Class::MultiSelectionType::C);
}

#include "PointPickerNode.h"
template <typename TDataType>
void declare_point_picker_node(py::module& m, std::string typestr) {
	using Class = dyno::PointPickerNode<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("PointPickerNode") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>PPN(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	PPN.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("resetStates", &Class::resetStates)

		.def("changePickingType", &Class::changePickingType)
		.def("changeMultiSelectionType", &Class::changeMultiSelectionType)

		.def("inTopology", &Class::inTopology, py::return_value_policy::reference)

		.def("statePointIndex", &Class::statePointIndex, py::return_value_policy::reference)

		.def("varPickingType", &Class::varPickingType, py::return_value_policy::reference)

		.def("varMultiSelectionType", &Class::varMultiSelectionType, py::return_value_policy::reference)

		.def("varInteractionRadius", &Class::varInteractionRadius, py::return_value_policy::reference)
		.def("varPointSelectedSize", &Class::varPointSelectedSize, py::return_value_policy::reference)
		.def("varPointOtherSize", &Class::varPointOtherSize, py::return_value_policy::reference)
		.def("varToggleIndexOutput", &Class::varToggleIndexOutput, py::return_value_policy::reference);

	py::enum_<typename Class::PickingTypeSelection>(PPN, "PickingTypeSelection")
		.value("Click", Class::PickingTypeSelection::Click)
		.value("Drag", Class::PickingTypeSelection::Drag)
		.value("Both", Class::PickingTypeSelection::Both);

	py::enum_<typename Class::MultiSelectionType>(PPN, "MultiSelectionType")
		.value("OR", Class::MultiSelectionType::OR)
		.value("XOR", Class::MultiSelectionType::XOR)
		.value("C", Class::MultiSelectionType::C);
}

#include "QuadPickerNode.h"
template <typename TDataType>
void declare_quad_picker_node(py::module& m, std::string typestr) {
	using Class = dyno::QuadPickerNode<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("QuadPickerNode") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>QPN(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	QPN.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("resetStates", &Class::resetStates)

		.def("changePickingElementType", &Class::changePickingElementType)
		.def("changePickingType", &Class::changePickingType)
		.def("changeMultiSelectionType", &Class::changeMultiSelectionType)

		.def("inTopology", &Class::inTopology, py::return_value_policy::reference)

		.def("stateQuadIndex", &Class::stateQuadIndex, py::return_value_policy::reference)
		.def("stateEdgeIndex", &Class::stateEdgeIndex, py::return_value_policy::reference)
		.def("statePointIndex", &Class::statePointIndex, py::return_value_policy::reference)
		.def("stateSur2PointIndex", &Class::stateSur2PointIndex, py::return_value_policy::reference)

		.def("varPickingElementType", &Class::varPickingElementType, py::return_value_policy::reference)

		.def("varPickingType", &Class::varPickingType, py::return_value_policy::reference)

		.def("varMultiSelectionType", &Class::varMultiSelectionType, py::return_value_policy::reference)

		.def("varToggleFlood", &Class::varToggleFlood, py::return_value_policy::reference)
		.def("varToggleVisibleFilter", &Class::varToggleVisibleFilter, py::return_value_policy::reference)
		.def("varToggleIndexOutput", &Class::varToggleIndexOutput, py::return_value_policy::reference)
		.def("varFloodAngle", &Class::varFloodAngle, py::return_value_policy::reference)
		.def("varInteractionRadius", &Class::varInteractionRadius, py::return_value_policy::reference)

		.def("varPointSelectedSize", &Class::varPointSelectedSize, py::return_value_policy::reference)
		.def("varPointOtherSize", &Class::varPointOtherSize, py::return_value_policy::reference)

		.def("varEdgeSelectedSize", &Class::varEdgeSelectedSize, py::return_value_policy::reference)
		.def("varEdgeOtherSize", &Class::varEdgeOtherSize, py::return_value_policy::reference);

	py::enum_<typename Class::PickingElementTypeSelection>(QPN, "PickingElementTypeSelection")
		.value("Edge", Class::PickingElementTypeSelection::Edge)
		.value("Point", Class::PickingElementTypeSelection::Point)
		.value("All", Class::PickingElementTypeSelection::All);

	py::enum_<typename Class::PickingTypeSelection>(QPN, "PickingTypeSelection")
		.value("Click", Class::PickingTypeSelection::Click)
		.value("Drag", Class::PickingTypeSelection::Drag)
		.value("Both", Class::PickingTypeSelection::Both);

	py::enum_<typename Class::MultiSelectionType>(QPN, "MultiSelectionType")
		.value("OR", Class::MultiSelectionType::OR)
		.value("XOR", Class::MultiSelectionType::XOR)
		.value("C", Class::MultiSelectionType::C);
}

#include "TrianglePickerNode.h"
template <typename TDataType>
void declare_triangle_picker_node(py::module& m, std::string typestr) {
	using Class = dyno::TrianglePickerNode<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("TrianglePickerNode") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>TPN(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	TPN.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("resetStates", &Class::resetStates)

		.def("changePickingElementType", &Class::changePickingElementType)
		.def("changePickingType", &Class::changePickingType)
		.def("changeMultiSelectionType", &Class::changeMultiSelectionType)

		.def("inTopology", &Class::inTopology, py::return_value_policy::reference)

		.def("stateTriQuadIndex", &Class::stateTriQuadIndex, py::return_value_policy::reference)
		.def("stateEdgeIndex", &Class::stateEdgeIndex, py::return_value_policy::reference)
		.def("statePointIndex", &Class::statePointIndex, py::return_value_policy::reference)
		.def("stateSur2PointIndex", &Class::stateSur2PointIndex, py::return_value_policy::reference)

		.def("varPickingElementType", &Class::varPickingElementType, py::return_value_policy::reference)

		.def("varPickingType", &Class::varPickingType, py::return_value_policy::reference)

		.def("varMultiSelectionType", &Class::varMultiSelectionType, py::return_value_policy::reference)

		.def("varToggleFlood", &Class::varToggleFlood, py::return_value_policy::reference)
		.def("varToggleVisibleFilter", &Class::varToggleVisibleFilter, py::return_value_policy::reference)
		.def("varToggleIndexOutput", &Class::varToggleIndexOutput, py::return_value_policy::reference)
		.def("varFloodAngle", &Class::varFloodAngle, py::return_value_policy::reference)
		.def("varInteractionRadius", &Class::varInteractionRadius, py::return_value_policy::reference)

		.def("varPointSelectedSize", &Class::varPointSelectedSize, py::return_value_policy::reference)
		.def("varPointOtherSize", &Class::varPointOtherSize, py::return_value_policy::reference)

		.def("varEdgeSelectedSize", &Class::varEdgeSelectedSize, py::return_value_policy::reference)
		.def("varEdgeOtherSize", &Class::varEdgeOtherSize, py::return_value_policy::reference);

	py::enum_<typename Class::PickingElementTypeSelection>(TPN, "PickingElementTypeSelection")
		.value("Edge", Class::PickingElementTypeSelection::Edge)
		.value("Point", Class::PickingElementTypeSelection::Point)
		.value("All", Class::PickingElementTypeSelection::All);

	py::enum_<typename Class::PickingTypeSelection>(TPN, "PickingTypeSelection")
		.value("Click", Class::PickingTypeSelection::Click)
		.value("Drag", Class::PickingTypeSelection::Drag)
		.value("Both", Class::PickingTypeSelection::Both);

	py::enum_<typename Class::MultiSelectionType>(TPN, "MultiSelectionType")
		.value("OR", Class::MultiSelectionType::OR)
		.value("XOR", Class::MultiSelectionType::XOR)
		.value("C", Class::MultiSelectionType::C);
}



void pybind_Interaction(py::module& m);