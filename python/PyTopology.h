#pragma once
#include "PyCommon.h"

#include "Collision/CalculateBoundingBox.h"
template<typename TDataType>
void declare_calculate_bounding_box(py::module& m, std::string typestr) {
	using Class = dyno::CalculateBoundingBox<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("CalculateBoundingBox") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("compute", &Class::compute)
		.def("inDiscreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("outAABB", &Class::outAABB, py::return_value_policy::reference);
}

#include "Collision/CollisionDetectionBroadPhase.h"
template<typename TDataType>
void declare_collision_detection_broad_phase(py::module& m, std::string typestr) {
	using Class = dyno::CollisionDetectionBroadPhase<TDataType>;
	using Parent = dyno::ComputeModule;

	class CollisionDetectionBroadPhaseTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CollisionDetectionBroadPhase<TDataType>,
				compute
			);
		}
	};

	class CollisionDetectionBroadPhasePublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("CollisionDetectionBroadPhase") + typestr;
	py::class_<Class, Parent, CollisionDetectionBroadPhaseTrampoline, std::shared_ptr<Class>>CDBP(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	CDBP.def(py::init<>())
		.def("varAccelerationStructure", &Class::varAccelerationStructure, py::return_value_policy::reference)
		.def("varGridSizeLimit", &Class::varGridSizeLimit, py::return_value_policy::reference)
		.def("varSelfCollision", &Class::varSelfCollision, py::return_value_policy::reference)
		.def("inSource", &Class::inSource, py::return_value_policy::reference)
		.def("inTarget", &Class::inTarget, py::return_value_policy::reference)
		.def("outContactList", &Class::outContactList, py::return_value_policy::reference)
		// protected
		.def("compute", &CollisionDetectionBroadPhasePublicist::compute);

	py::enum_<typename Class::EStructure>(CDBP, "EStructure")
		.value("BVH", Class::EStructure::BVH)
		.value("Octree", Class::EStructure::Octree)
		.export_values();
}

#include "Collision/CollistionDetectionBoundingBox.h"
template<typename TDataType>
void declare_collistion_detection_bounding_box(py::module& m, std::string typestr) {
	using Class = dyno::CollistionDetectionBoundingBox<TDataType>;
	using Parent = dyno::ComputeModule;

	class CollistionDetectionBoundingBoxTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CollistionDetectionBoundingBox<TDataType>,
				compute
			);
		}
	};

	class CollistionDetectionBoundingBoxPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("CollistionDetectionBoundingBox") + typestr;
	py::class_<Class, Parent, CollistionDetectionBoundingBoxTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varUpperBound", &Class::varUpperBound, py::return_value_policy::reference)
		.def("varLowerBound", &Class::varLowerBound, py::return_value_policy::reference)
		.def("inDiscreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("outContacts", &Class::outContacts, py::return_value_policy::reference)
		// protected
		.def("compute", &CollistionDetectionBoundingBoxPublicist::compute);
}

#include "Collision/CollistionDetectionTriangleSet.h"
template<typename TDataType>
void declare_collistion_detection_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::CollistionDetectionTriangleSet<TDataType>;
	using Parent = dyno::ComputeModule;

	class CollistionDetectionTriangleSetTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CollistionDetectionTriangleSet<TDataType>,
				compute
			);
		}
	};

	class CollistionDetectionTriangleSetPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("CollistionDetectionTriangleSet") + typestr;
	py::class_<Class, Parent, CollistionDetectionTriangleSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inDiscreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("outContacts", &Class::outContacts, py::return_value_policy::reference)
		// protected
		.def("compute", &CollistionDetectionTriangleSetPublicist::compute);
}

#include "Collision/NeighborElementQuery.h"
template<typename TDataType>
void declare_neighbor_element_query(py::module& m, std::string typestr) {
	using Class = dyno::NeighborElementQuery<TDataType>;
	using Parent = dyno::ComputeModule;

	class NeighborElementQueryTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::NeighborElementQuery<TDataType>,
				compute
			);
		}
	};

	class NeighborElementQueryPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("NeighborElementQuery") + typestr;
	py::class_<Class, Parent, NeighborElementQueryTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSelfCollision", &Class::varSelfCollision, py::return_value_policy::reference)
		.def("varDHead", &Class::varDHead, py::return_value_policy::reference)

		.def("varGridSizeLimit", &Class::varGridSizeLimit, py::return_value_policy::reference)
		.def("inDiscreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("inCollisionMask", &Class::inCollisionMask, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("outContacts", &Class::outContacts, py::return_value_policy::reference)
		// protected
		.def("compute", &NeighborElementQueryPublicist::compute);
}

#include "Collision/NeighborPointQuery.h"
template<typename TDataType>
void declare_neighbor_point_query(py::module& m, std::string typestr) {
	using Class = dyno::NeighborPointQuery<TDataType>;
	using Parent = dyno::ComputeModule;

	class NeighborPointQueryTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::NeighborPointQuery<TDataType>,
				compute
			);
		}
	};

	class NeighborPointQueryPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("NeighborPointQuery") + typestr;
	py::class_<Class, Parent, NeighborPointQueryTrampoline,  std::shared_ptr<Class>>NPQ(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	NPQ.def(py::init<>())
		.def("varSpatial", &Class::varSpatial, py::return_value_policy::reference)
		.def("varSizeLimit", &Class::varSizeLimit, py::return_value_policy::reference)
		.def("inRadius", &Class::inRadius, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inOther", &Class::inOther, py::return_value_policy::reference)
		.def("outNeighborIds", &Class::outNeighborIds, py::return_value_policy::reference)
		// protected
		.def("compute", &NeighborPointQueryPublicist::compute);

	py::enum_<typename Class::Spatial>(NPQ, "Spatial")
		.value("UNIFORM", Class::Spatial::UNIFORM)
		.value("BVH", Class::Spatial::BVH)
		.value("OCTREE", Class::Spatial::OCTREE)
		.export_values();
}

#include "Collision/NeighborTriangleQuery.h"
template<typename TDataType>
void declare_neighbor_triangle_query(py::module& m, std::string typestr) {
	using Class = dyno::NeighborTriangleQuery<TDataType>;
	using Parent = dyno::ComputeModule;

	class NeighborTriangleQueryTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::NeighborTriangleQuery<TDataType>,
				compute
			);
		}
	};

	class NeighborTriangleQueryPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("NeighborTriangleQuery") + typestr;
	py::class_<Class, Parent, NeighborTriangleQueryTrampoline, std::shared_ptr<Class>>NTQ(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	NTQ.def(py::init<>())
		.def("varSpatial", &Class::varSpatial, py::return_value_policy::reference)
		.def("inRadius", &Class::inRadius, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("outNeighborIds", &Class::outNeighborIds, py::return_value_policy::reference)
		// protected
		.def("compute", &NeighborTriangleQueryPublicist::compute);

	py::enum_<typename Class::Spatial>(NTQ, "Spatial")
		.value("BVH", Class::Spatial::BVH)
		.value("OCTREE", Class::Spatial::OCTREE);
}

#include "Mapping/AnchorPointToPointSet.h"
template <typename TDataType>
void declare_anchor_point_to_point_set(py::module& m, std::string typestr) {
	using Class = dyno::AnchorPointToPointSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class AnchorPointToPointSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::AnchorPointToPointSet<TDataType>,
				apply
			);
		}
	};

	class AnchorPointToPointSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("AnchorPointToPointSet") + typestr;
	py::class_<Class, Parent, AnchorPointToPointSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inDiscreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("inCenter", &Class::inCenter, py::return_value_policy::reference)
		.def("inRotationMatrix", &Class::inRotationMatrix, py::return_value_policy::reference)
		.def("outPointSet", &Class::outPointSet, py::return_value_policy::reference)
		// protected
		.def("apply", &AnchorPointToPointSetPublicist::apply);
}

#include "Mapping/BoundingBoxToEdgeSet.h"
template <typename TDataType>
void declare_bounding_box_to_edge_set(py::module& m, std::string typestr) {
	using Class = dyno::BoundingBoxToEdgeSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class BoundingBoxToEdgeSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::BoundingBoxToEdgeSet<TDataType>,
				apply
			);
		}
	};

	class BoundingBoxToEdgeSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("BoundingBoxToEdgeSet") + typestr;
	py::class_<Class, Parent, BoundingBoxToEdgeSetTrampoline,  std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inAABB", &Class::inAABB, py::return_value_policy::reference)
		.def("outEdgeSet", &Class::outEdgeSet, py::return_value_policy::reference)
		// protected
		.def("apply", &BoundingBoxToEdgeSetPublicist::apply);
}

#include "Mapping/ContactsToEdgeSet.h"
template<typename TDataType>
void declare_contacts_to_edge_set(py::module& m, std::string typestr) {
	using Class = dyno::ContactsToEdgeSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class ContactsToEdgeSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::ContactsToEdgeSet<TDataType>,
				apply
			);
		}
	};

	class ContactsToEdgeSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("ContactsToEdgeSet") + typestr;
	py::class_<Class, Parent, ContactsToEdgeSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varScale", &Class::varScale, py::return_value_policy::reference)
		.def("inContacts", &Class::inContacts, py::return_value_policy::reference)
		.def("outEdgeSet", &Class::outEdgeSet, py::return_value_policy::reference)
		// protected
		.def("apply", &ContactsToEdgeSetPublicist::apply);
}

#include "Mapping/ContactsToPointSet.h"
template<typename TDataType>
void declare_contacts_to_point_set(py::module& m, std::string typestr) {
	using Class = dyno::ContactsToPointSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class ContactsToPointSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::ContactsToPointSet<TDataType>,
				apply
			);
		}
	};

	class ContactsToPointSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("ContactsToPointSet") + typestr;
	py::class_<Class, Parent, ContactsToPointSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inContacts", &Class::inContacts, py::return_value_policy::reference)
		.def("outPointSet", &Class::outPointSet, py::return_value_policy::reference)
		// protected
		.def("apply", &ContactsToPointSetPublicist::apply);
}

#include "Mapping/DiscreteElementsToTriangleSet.h"
template <typename TDataType>
void declare_discrete_elements_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::DiscreteElementsToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class DiscreteElementsToTriangleSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::DiscreteElementsToTriangleSet<TDataType>,
				apply
			);
		}
	};

	class DiscreteElementsToTriangleSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("DiscreteElementsToTriangleSet") + typestr;
	py::class_<Class, Parent, DiscreteElementsToTriangleSetTrampoline,std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inDiscreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference)
		// protected
		.def("apply", &DiscreteElementsToTriangleSetPublicist::apply);
}

#include "Mapping/Extract.h"
template <typename TDataType>
void declare_extract_edge_set_from_polygon_set(py::module& m, std::string typestr) {
	using Class = dyno::ExtractEdgeSetFromPolygonSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class ExtractEdgeSetFromPolygonSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::ExtractEdgeSetFromPolygonSet<TDataType>,
				apply
			);
		}
	};

	class ExtractEdgeSetFromPolygonSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("ExtractEdgeSetFromPolygonSet") + typestr;
	py::class_<Class, Parent, ExtractEdgeSetFromPolygonSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("inPolygonSet", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("outEdgeSet", &Class::outEdgeSet, py::return_value_policy::reference)
		// protected
		.def("apply", &ExtractEdgeSetFromPolygonSetPublicist::apply);
}

template <typename TDataType>
void declare_extract_triangle_set_from_polygon_set(py::module& m, std::string typestr) {
	using Class = dyno::ExtractTriangleSetFromPolygonSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class ExtractTriangleSetFromPolygonSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::ExtractTriangleSetFromPolygonSet<TDataType>,
				apply
			);
		}
	};

	class ExtractTriangleSetFromPolygonSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("ExtractTriangleSetFromPolygonSet") + typestr;
	py::class_<Class, Parent, ExtractTriangleSetFromPolygonSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("inPolygonSet", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference)
		// protected
		.def("apply", &ExtractTriangleSetFromPolygonSetPublicist::apply);
}

template <typename TDataType>
void declare_extract_qaud_set_from_polygon_set(py::module& m, std::string typestr) {
	using Class = dyno::ExtractQaudSetFromPolygonSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class ExtractQaudSetFromPolygonSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::ExtractQaudSetFromPolygonSet<TDataType>,
				apply
			);
		}
	};

	class ExtractQaudSetFromPolygonSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("ExtractQaudSetFromPolygonSet") + typestr;
	py::class_<Class, Parent, ExtractQaudSetFromPolygonSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("inPolygonSet", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("outQuadSet", &Class::outQuadSet, py::return_value_policy::reference)
		// protected
		.def("apply", &ExtractQaudSetFromPolygonSetPublicist::apply);
}

#include "Mapping/FrameToPointSet.h"
template <typename TDataType>
void declare_frame_to_point_set(py::module& m, std::string typestr) {
	using Class = dyno::FrameToPointSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class FrameToPointSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool initializeImpl() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::FrameToPointSet<TDataType>,
				initializeImpl
			);
		}
	};

	class FrameToPointSetPublicist : public Class
	{
	public:
		using Class::initializeImpl;
	};

	std::string pyclass_name = std::string("FrameToPointSet") + typestr;
	py::class_<Class, Parent, FrameToPointSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("initialize", &Class::initialize)
		.def("applyTransform", &Class::applyTransform)
		.def("apply", &Class::apply)
		// protected
		.def("initializeImpl", &FrameToPointSetPublicist::initializeImpl);
}

#include "Mapping/HeightFieldToTriangleSet.h"
template <typename TDataType>
void declare_height_field_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::HeightFieldToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class HeightFieldToTriangleSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::HeightFieldToTriangleSet<TDataType>,
				apply
			);
		}
	};

	class HeightFieldToTriangleSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("HeightFieldToTriangleSet") + typestr;
	py::class_<Class, Parent, HeightFieldToTriangleSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inHeightField", &Class::inHeightField, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference)
		.def("varScale", &Class::varScale, py::return_value_policy::reference)
		.def("varTranslation", &Class::varTranslation, py::return_value_policy::reference)
		// protected
		.def("apply", &HeightFieldToTriangleSetPublicist::apply);
}

#include "Mapping/MergeSimplexSet.h"
template <typename TDataType>
void declare_merge_simplex_set(py::module& m, std::string typestr) {
	using Class = dyno::MergeSimplexSet<TDataType>;
	using Parent = dyno::Node;

	class MergeSimplexSetTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MergeSimplexSet<TDataType>,
				resetStates
			);
		}
	};

	class MergeSimplexSetPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("MergeSimplexSet") + typestr;
	py::class_<Class, Parent, MergeSimplexSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inEdgeSet", &Class::inEdgeSet, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("inTetrahedronSet", &Class::inTetrahedronSet, py::return_value_policy::reference)
		.def("outSimplexSet", &Class::outSimplexSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &MergeSimplexSetPublicist::resetStates);
}

#include "Mapping/MergeTriangleSet.h"
template <typename TDataType>
void declare_merge_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::MergeTriangleSet<TDataType>;
	using Parent = dyno::Node;

	class MergeTriangleSetTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MergeTriangleSet<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MergeTriangleSet<TDataType>,
				updateStates
			);
		}
	};

	class MergeTriangleSetPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("MergeTriangleSet") + typestr;
	py::class_<Class, Parent, MergeTriangleSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("inFirst", &Class::inFirst, py::return_value_policy::reference)
		.def("inSecond", &Class::inSecond, py::return_value_policy::reference)
		// protected
		.def("resetStates", &MergeTriangleSetPublicist::resetStates)
		.def("updateStates", &MergeTriangleSetPublicist::updateStates);
}

#include "Mapping/PointSetToPointSet.h"
template <typename TDataType>
void declare_point_set_to_point_set(py::module& m, std::string typestr) {
	using Class = dyno::PointSetToPointSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class PointSetToPointSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool initializeImpl() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::PointSetToPointSet<TDataType>,
				initializeImpl
			);
		}
	};

	class PointSetToPointSetPublicist : public Class
	{
	public:
		using Class::initializeImpl;
	};

	std::string pyclass_name = std::string("PointSetToPointSet") + typestr;
	py::class_<Class, Parent, PointSetToPointSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<std::shared_ptr<dyno::PointSet<TDataType>>, std::shared_ptr<dyno::PointSet<TDataType>> >())
		.def("setSearchingRadius", &Class::setSearchingRadius)
		.def("setFrom", &Class::setFrom)
		.def("setTo", &Class::setTo)
		.def("apply", &Class::apply)
		.def("match", &Class::match)
		// protected
		.def("initializeImpl", &PointSetToPointSetPublicist::initializeImpl);
}

#include "Mapping/PointSetToTriangleSet.h"
template <typename TDataType>
void declare_point_set_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::PointSetToTriangleSet<TDataType>;
	using Parent = dyno::Node;

	class PointSetToTriangleSetTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PointSetToTriangleSet<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PointSetToTriangleSet<TDataType>,
				updateStates
			);
		}
	};

	class PointSetToTriangleSetPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("PointSetToTriangleSet") + typestr;
	py::class_<Class, Parent, PointSetToTriangleSetTrampoline,std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inPointSet", &Class::inPointSet, py::return_value_policy::reference)
		.def("inInitialShape", &Class::inInitialShape, py::return_value_policy::reference)
		.def("outShape", &Class::outShape, py::return_value_policy::reference)
		// protected
		.def("resetStates", &PointSetToTriangleSetPublicist::resetStates)
		.def("updateStates", &PointSetToTriangleSetPublicist::updateStates);
}

#include "Mapping/QuadSetToTriangleSet.h"
template <typename TDataType>
void declare_quad_set_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::QuadSetToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class QuadSetToTriangleSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::QuadSetToTriangleSet<TDataType>,
				apply
			);
		}
	};

	class QuadSetToTriangleSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("QuadSetToTriangleSet") + typestr;
	py::class_<Class, Parent, QuadSetToTriangleSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inQuadSet", &Class::inQuadSet, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference)
		// protected
		.def("apply", &QuadSetToTriangleSetPublicist::apply);
}

#include "Mapping/SplitSimplexSet.h"
template <typename TDataType>
void declare_split_simplex_set(py::module& m, std::string typestr) {
	using Class = dyno::SplitSimplexSet<TDataType>;
	using Parent = dyno::Node;

	class SplitSimplexSetTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SplitSimplexSet<TDataType>,
				resetStates
			);
		}

	};

	class SplitSimplexSetPublicist : public Class
	{
	public:
		using Class::resetStates;
	};


	std::string pyclass_name = std::string("SplitSimplexSet") + typestr;
	py::class_<Class, Parent, SplitSimplexSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inSimplexSet", &Class::inSimplexSet, py::return_value_policy::reference)
		.def("outEdgeSet", &Class::outEdgeSet, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference)
		.def("outTetrahedronSet", &Class::outTetrahedronSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &SplitSimplexSetPublicist::resetStates);
}

#include "Mapping/TetrahedronSetToPointSet.h"
template <typename TDataType>
void declare_tetrahedron_set_to_point_set(py::module& m, std::string typestr) {
	using Class = dyno::TetrahedronSetToPointSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class TetrahedronSetToPointSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::TetrahedronSetToPointSet<TDataType>,
				apply
			);
		}

		bool initializeImpl() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::TetrahedronSetToPointSet<TDataType>,
				initializeImpl
			);
		}
	};

	class TetrahedronSetToPointSetPublicist : public Class
	{
	public:
		using Class::apply;
		using Class::initializeImpl;
	};

	std::string pyclass_name = std::string("TetrahedronSetToPointSet") + typestr;
	py::class_<Class, Parent, TetrahedronSetToPointSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("setSearchingRadius", &Class::setSearchingRadius)
		.def("setFrom", &Class::setFrom)
		.def("setTo", &Class::setTo)
		.def("match", &Class::match)
		// protected
		.def("apply", &TetrahedronSetToPointSetPublicist::apply)
		.def("initializeImpl", &TetrahedronSetToPointSetPublicist::initializeImpl);
}

#include "Mapping/TextureMeshToTriangleSet.h"
template <typename TDataType>
void declare_texture_mesh_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::TextureMeshToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class TextureMeshToTriangleSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::TextureMeshToTriangleSet<TDataType>,
				apply
			);
		}
	};

	class TextureMeshToTriangleSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("TextureMeshToTriangleSet") + typestr;
	py::class_<Class, Parent, TextureMeshToTriangleSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTextureMesh", &Class::inTextureMesh, py::return_value_policy::reference)
		.def("inTransform", &Class::inTransform, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference)
		// protected
		.def("apply", &TextureMeshToTriangleSetPublicist::apply);
}

template <typename TDataType>
void declare_texture_mesh_to_triangle_set_node(py::module& m, std::string typestr) {
	using Class = dyno::TextureMeshToTriangleSetNode<TDataType>;
	using Parent = dyno::Node;

	class TextureMeshToTriangleSetNodeTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TextureMeshToTriangleSetNode<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TextureMeshToTriangleSetNode<TDataType>,
				updateStates
			);
		}
	};

	class TextureMeshToTriangleSetNodePublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("TextureMeshToTriangleSetNode") + typestr;
	py::class_<Class, Parent, TextureMeshToTriangleSetNodeTrampoline,std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("inTextureMesh", &Class::inTextureMesh, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &TextureMeshToTriangleSetNodePublicist::resetStates)
		.def("updateStates", &TextureMeshToTriangleSetNodePublicist::updateStates);
}

#include "Module/CalculateMaximum.h"
template <typename TDataType>
void declare_calculate_maximum(py::module& m, std::string typestr) {
	using Class = dyno::CalculateMaximum<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("CalculateMaximum") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("compute", &Class::compute)
		.def("inScalarArray", &Class::inScalarArray, py::return_value_policy::reference)
		.def("outScalar", &Class::outScalar, py::return_value_policy::reference);
}

#include "Module/CalculateMinimum.h"
template <typename TDataType>
void declare_calculate_minimum(py::module& m, std::string typestr) {
	using Class = dyno::CalculateMinimum<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("CalculateMinimum") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("compute", &Class::compute)
		.def("inScalarArray", &Class::inScalarArray, py::return_value_policy::reference)
		.def("outScalar", &Class::outScalar, py::return_value_policy::reference);
}

#include "Module/CalculateNorm.h"
template <typename TDataType>
void declare_calculate_norm(py::module& m, std::string typestr) {
	using Class = dyno::CalculateNorm<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("CalculateNorm") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("compute", &Class::compute)
		.def("inVec", &Class::inVec, py::return_value_policy::reference)
		.def("outNorm", &Class::outNorm, py::return_value_policy::reference);
}

#include "Topology/AnimationCurve.h"
template <typename TDataType>
void declare_animation_curve(py::module& m, std::string typestr) {
	using Class = dyno::AnimationCurve<TDataType>;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename ::dyno::Mat4f Mat;
	std::string pyclass_name = std::string("AnimationCurve") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<int, Real, Real, Real>())
		.def("set", &Class::set)
		.def("fbxTimeToSeconds", &Class::fbxTimeToSeconds)
		.def("secondsToFbxTime", &Class::secondsToFbxTime)
		.def("setInitVal", &Class::setInitVal)
		.def("getCurveValueAlong", &Class::getCurveValueAlong)
		.def("getCurveValueAll", &Class::getCurveValueAll)
		.def("getCurveValueCycle", &Class::getCurveValueCycle)

		.def_readwrite("m_maxSize", &Class::m_maxSize);
}

#include "Topology/DiscreteElements.h"
template<typename Real>
void declare_joint(py::module& m, std::string typestr) {
	using Class = dyno::Joint<Real>;
	std::string pyclass_name = std::string("Joint") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<dyno::PdActor*, dyno::PdActor*>())
		.def_readwrite("bodyId1", &Class::bodyId1)
		.def_readwrite("bodyId2", &Class::bodyId2)
		.def_readwrite("bodyType1", &Class::bodyType1)
		.def_readwrite("bodyType2", &Class::bodyType2)
		.def_readwrite("actor1", &Class::actor1)
		.def_readwrite("actor2", &Class::actor2);
}

template<typename Real>
void declare_ball_and_socket_joint(py::module& m, std::string typestr) {
	using Class = dyno::BallAndSocketJoint<Real>;
	using Parent = dyno::Joint<Real>;
	std::string pyclass_name = std::string("BallAndSocketJoint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<dyno::PdActor*, dyno::PdActor*>())
		.def("setAnchorPoint", &Class::setAnchorPoint)
		.def_readwrite("r1", &Class::r1)
		.def_readwrite("r2", &Class::r2);
}

template<typename Real>
void declare_slider_joint(py::module& m, std::string typestr) {
	using Class = dyno::SliderJoint<Real>;
	using Parent = dyno::Joint<Real>;
	std::string pyclass_name = std::string("SliderJoint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<dyno::PdActor*, dyno::PdActor*>())
		.def("setAnchorPoint", &Class::setAnchorPoint)
		.def("setAxis", &Class::setAxis)
		.def("setMoter", &Class::setMoter)
		.def("setRange", &Class::setRange)

		.def_readwrite("useRange", &Class::useRange)
		.def_readwrite("useMoter", &Class::useMoter)
		.def_readwrite("d_min", &Class::d_min)
		.def_readwrite("d_max", &Class::d_max)
		.def_readwrite("v_moter", &Class::v_moter)
		.def_readwrite("r1", &Class::r1)
		.def_readwrite("r2", &Class::r2)
		.def_readwrite("sliderAxis", &Class::sliderAxis)
		.def_readwrite("q_init", &Class::q_init);
}

template<typename Real>
void declare_hinge_joint(py::module& m, std::string typestr) {
	using Class = dyno::HingeJoint<Real>;
	using Parent = dyno::Joint<Real>;
	std::string pyclass_name = std::string("HingeJoint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<dyno::PdActor*, dyno::PdActor*>())
		.def("setAnchorPoint", &Class::setAnchorPoint)
		.def("setAxis", &Class::setAxis)
		.def("setMoter", &Class::setMoter)
		.def("setRange", &Class::setRange)

		.def_readwrite("d_min", &Class::d_min)
		.def_readwrite("d_max", &Class::d_max)
		.def_readwrite("v_moter", &Class::v_moter)
		.def_readwrite("r1", &Class::r1)
		.def_readwrite("r2", &Class::r2)
		.def_readwrite("hingeAxisBody1", &Class::hingeAxisBody1)
		.def_readwrite("hingeAxisBody2", &Class::hingeAxisBody2)
		.def_readwrite("useRange", &Class::useRange)
		.def_readwrite("useMoter", &Class::useMoter);
}

template<typename Real>
void declare_fixed_joint(py::module& m, std::string typestr) {
	using Class = dyno::FixedJoint<Real>;
	using Parent = dyno::Joint<Real>;
	std::string pyclass_name = std::string("FixedJoint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<dyno::PdActor*, dyno::PdActor*>())
		.def(py::init<dyno::PdActor*>())
		.def("setAnchorPoint", &Class::setAnchorPoint)
		.def("setAnchorAngle", &Class::setAnchorAngle)

		.def_readwrite("r1", &Class::r1)
		.def_readwrite("r2", &Class::r2)
		.def_readwrite("w", &Class::w)
		.def_readwrite("q", &Class::q)
		.def_readwrite("q_init", &Class::q_init);
}

template<typename Real>
void declare_point_joint(py::module& m, std::string typestr) {
	using Class = dyno::PointJoint<Real>;
	using Parent = dyno::Joint<Real>;
	std::string pyclass_name = std::string("PointJoint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<dyno::PdActor*>())
		.def("setAnchorPoint", &Class::setAnchorPoint)

		.def_readwrite("anchorPoint", &Class::anchorPoint);
}

template<typename Real>
void declare_distance_joint(py::module& m, std::string typestr) {
	using Class = dyno::DistanceJoint<Real>;
	using Parent = dyno::Joint<Real>;
	std::string pyclass_name = std::string("DistanceJoint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<dyno::PdActor*, dyno::PdActor*>())
		.def("setDistanceJoint", &Class::setDistanceJoint)

		.def_readwrite("r1", &Class::r1)
		.def_readwrite("r2", &Class::r2)
		.def_readwrite("distance", &Class::distance);
}

template <typename TDataType>
void declare_discrete_elements(py::module& m, std::string typestr) {
	using Class = dyno::DiscreteElements<TDataType>;
	using Parent = dyno::TopologyModule;

	class DiscreteElementsTrampoline : public Class
	{
	public:
		using Class::Class;

		void updateTopology() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::DiscreteElements<TDataType>,
				updateTopology
			);
		}
	};

	class DiscreteElementsPublicist : public Class
	{
	public:
		using Class::updateTopology;
	};

	std::string pyclass_name = std::string("DiscreteElements") + typestr;
	py::class_<Class, Parent, DiscreteElementsTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("scale", &Class::scale)

		.def("totalSize", &Class::totalSize)

		.def("totalJointSize", &Class::totalJointSize)

		.def("sphereIndex", &Class::sphereIndex)
		.def("boxIndex", &Class::boxIndex)
		.def("capsuleIndex", &Class::capsuleIndex)
		.def("tetIndex", &Class::tetIndex)
		.def("triangleIndex", &Class::triangleIndex)

		.def("calculateElementOffset", &Class::calculateElementOffset)

		.def("setSpheres", &Class::setSpheres)
		.def("setBoxes", &Class::setBoxes)
		.def("setTets", &Class::setTets)
		.def("setCapsules", &Class::setCapsules)
		.def("setTriangles", &Class::setTriangles)
		.def("setTetSDF", &Class::setTetSDF)

		.def("spheresInLocal", &Class::spheresInLocal, py::return_value_policy::reference)
		.def("boxesInLocal", &Class::boxesInLocal, py::return_value_policy::reference)
		.def("tetsInLocal", &Class::tetsInLocal, py::return_value_policy::reference)
		.def("capsulesInLocal", &Class::capsulesInLocal, py::return_value_policy::reference)
		.def("trianglesInLocal", &Class::trianglesInLocal, py::return_value_policy::reference)

		.def("spheresInGlobal", &Class::spheresInGlobal, py::return_value_policy::reference)
		.def("boxesInGlobal", &Class::boxesInGlobal, py::return_value_policy::reference)
		.def("tetsInGlobal", &Class::tetsInGlobal, py::return_value_policy::reference)
		.def("capsulesInGlobal", &Class::capsulesInGlobal, py::return_value_policy::reference)
		.def("trianglesInGlobal", &Class::trianglesInGlobal, py::return_value_policy::reference)

		.def("shape2RigidBodyMapping", &Class::shape2RigidBodyMapping, py::return_value_policy::reference)

		.def("position", &Class::position, py::return_value_policy::reference)
		.def("rotation", &Class::rotation, py::return_value_policy::reference)

		.def("setPosition", &Class::setPosition)
		.def("setRotation", &Class::setRotation)

		.def("ballAndSocketJoints", &Class::ballAndSocketJoints, py::return_value_policy::reference)
		.def("sliderJoints", &Class::sliderJoints, py::return_value_policy::reference)
		.def("hingeJoints", &Class::hingeJoints, py::return_value_policy::reference)
		.def("fixedJoints", &Class::fixedJoints, py::return_value_policy::reference)
		.def("pointJoints", &Class::pointJoints, py::return_value_policy::reference)
		.def("distanceJoints", &Class::distanceJoints, py::return_value_policy::reference)

		.def("setTetBodyId", &Class::setTetBodyId)
		.def("setTetElementId", &Class::setTetElementId)

		.def("getTetSDF", &Class::getTetSDF, py::return_value_policy::reference)
		.def("getTetBodyMapping", &Class::getTetBodyMapping, py::return_value_policy::reference)
		.def("getTetElementMapping", &Class::getTetElementMapping, py::return_value_policy::reference)

		.def("copyFrom", &Class::copyFrom)
		.def("merge", &Class::merge)
		.def("requestDiscreteElementsInGlobal", &Class::requestDiscreteElementsInGlobal)

		.def("requestBoxInGlobal", &Class::requestBoxInGlobal, py::return_value_policy::reference)
		.def("requestSphereInGlobal", &Class::requestSphereInGlobal, py::return_value_policy::reference)
		.def("requestTetInGlobal", &Class::requestTetInGlobal, py::return_value_policy::reference)
		.def("requestCapsuleInGlobal", &Class::requestCapsuleInGlobal, py::return_value_policy::reference)
		.def("requestTriangleInGlobal", &Class::requestTriangleInGlobal, py::return_value_policy::reference)
		// protected
		.def("updateTopology", &DiscreteElementsPublicist::updateTopology);
}

#include "Topology/DistanceField3D.h"
template <typename TDataType>
void declare_distance_field3D(py::module& m, std::string typestr) {
	using Class = dyno::DistanceField3D<TDataType>;
	std::string pyclass_name = std::string("DistanceField3D") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("release", &Class::release)
		.def("translate", &Class::translate)
		.def("scale", &Class::scale)
		.def("getDistance", &Class::getDistance)
		.def("nx", &Class::nx)
		.def("ny", &Class::ny)
		.def("nz", &Class::nz)
		.def("loadSDF", &Class::loadSDF)
		.def("loadBox", &Class::loadBox)
		.def("loadCylinder", &Class::loadCylinder)
		.def("loadSphere", &Class::loadSphere)
		.def("setSpace", &Class::setSpace)
		.def("lowerBound", &Class::lowerBound)
		.def("upperBound", &Class::upperBound)
		.def("assign", &Class::assign)
		.def("get_m_distance", &Class::distances, py::return_value_policy::reference)
		.def("setDistance", &Class::setDistance)
		.def("get_h", &Class::getGridSpacing)
		.def("invertSDF", &Class::invertSDF);
}

#include "Topology/PointSet.h"
template <typename TDataType>
void declare_point_set(py::module& m, std::string typestr) {
	using Class = dyno::PointSet<TDataType>;
	using Parent = dyno::TopologyModule;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	std::string pyclass_name = std::string("PointSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("copyFrom", &Class::copyFrom)
		.def("set_points", py::overload_cast<const std::vector<Coord>&>(&Class::setPoints))
		.def("set_points", py::overload_cast<const dyno::Array<Coord, DeviceType::GPU>&>(&Class::setPoints))
		.def("setSize", &Class::setSize)

		.def("getPointSize", &Class::getPointSize)

		.def("requestBoundingBox", &Class::requestBoundingBox)
		.def("scale", py::overload_cast<const Real>(&Class::scale))
		.def("scale", py::overload_cast<const Coord>(&Class::scale))
		.def("translate", &Class::translate)

		.def("rotate", py::overload_cast<const Coord>(&Class::rotate))
		.def("rotate", py::overload_cast<const dyno::Quat<Real>>(&Class::rotate))

		.def("loadObjFile", &Class::loadObjFile)
		.def("isEmpty", &Class::isEmpty)
		.def("clear", &Class::clear)
		.def("getPoints", &Class::getPoints);
}

#include "Topology/EdgeSet.h"
template <typename TDataType>
void declare_edge_set(py::module& m, std::string typestr) {
	using Class = dyno::EdgeSet<TDataType>;
	using Parent = dyno::PointSet<TDataType>;
	typedef typename dyno::TopologyModule::Edge Edge;

	class EdgeSetTrampoline : public Class
	{
	public:
		using Class::Class;

		void updateTopology() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::EdgeSet<TDataType>,
				updateTopology
			);
		}
	};

	class EdgeSetPublicist : public Class
	{
	public:
		using Class::updateTopology;
		using Class::updateEdges;
		using Class::updateVer2Edge;
	};

	std::string pyclass_name = std::string("EdgeSet") + typestr;
	py::class_<Class, Parent, EdgeSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_edges", py::overload_cast<std::vector<Edge>&>(&Class::setEdges))
		.def("set_edges", py::overload_cast<dyno::Array<Edge, DeviceType::GPU>&>(&Class::setEdges))
		.def("edgeIndices", &Class::edgeIndices)
		.def("vertex2Edge", &Class::vertex2Edge)
		.def("copyFrom", &Class::copyFrom)
		.def("isEmpty", &Class::isEmpty)
		.def("clear", &Class::clear)
		.def("loadSmeshFile", &Class::loadSmeshFile)

		.def("requestPointNeighbors", &Class::requestPointNeighbors)
		// protected
		.def("updateTopology", &EdgeSetPublicist::updateTopology)
		.def("updateEdges", &EdgeSetPublicist::updateEdges)
		.def("updateVer2Edge", &EdgeSetPublicist::updateVer2Edge);
}

#include "Topology/Frame.h"
template <typename TDataType>
void declare_frame(py::module& m, std::string typestr) {
	using Class = dyno::Frame<TDataType>;
	using Parent = dyno::TopologyModule;
	std::string pyclass_name = std::string("Frame") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("copyFrom", &Class::copyFrom)
		.def("setCenter", &Class::setCenter)
		.def("getCenter", &Class::getCenter)
		.def("setOrientation", &Class::setOrientation)
		.def("getOrientation", &Class::getOrientation);
}

#include "Topology/GridHash.h"
template <typename TDataType>
void declare_grid_hash(py::module& m, std::string typestr) {
	using Class = dyno::GridHash<TDataType>;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	std::string pyclass_name = std::string("GridHash") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("setSpace", &Class::setSpace)
		.def("construct", &Class::construct)
		.def("clear", &Class::clear)
		.def("release", &Class::release)
		.def("get_index", py::overload_cast<int, int, int>(&Class::getIndex))
		.def("get_index", py::overload_cast<Coord>(&Class::getIndex))
		.def("getIndex3", &Class::getIndex3)
		.def("getCounter", &Class::getCounter)
		.def("getParticleId", &Class::getParticleId)

		.def_readwrite("num", &Class::num)
		.def_readwrite("nx", &Class::nx)
		.def_readwrite("ny", &Class::ny)
		.def_readwrite("nz", &Class::nz)
		.def_readwrite("particle_num", &Class::particle_num)
		.def_readwrite("ds", &Class::ds)
		.def_readwrite("lo", &Class::lo)
		.def_readwrite("hi", &Class::hi)
		.def_readwrite("ids", &Class::ids)
		.def_readwrite("counter", &Class::counter)
		.def_readwrite("index", &Class::index)
		.def_readwrite("m_scan", &Class::m_scan)
		.def_readwrite("m_reduce", &Class::m_reduce);
}

#include "Topology/GridSet.h"
template <typename TDataType>
void declare_grid_set(py::module& m, std::string typestr) {
	using Class = dyno::GridSet<TDataType>;
	using Parent = dyno::TopologyModule;
	std::string pyclass_name = std::string("GridSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("setUniGrid", &Class::setUniGrid)
		.def("setNijk", &Class::setNijk)
		.def("setOrigin", &Class::setOrigin)
		.def("setDx", &Class::setDx)

		.def("getNi", &Class::getNi)
		.def("getNj", &Class::getNj)
		.def("getNk", &Class::getNk)
		.def("getOrigin", &Class::getOrigin)
		.def("getDx", &Class::getDx);
}

#include "Topology/HeightField.h"
template <typename TDataType>
void declare_height_field(py::module& m, std::string typestr) {
	using Class = dyno::HeightField<TDataType>;
	using Parent = dyno::TopologyModule;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	std::string pyclass_name = std::string("HeightField") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("copyFrom", &Class::copyFrom)

		.def("scale", py::overload_cast<Real>(&Class::scale))
		.def("scale", py::overload_cast<Coord>(&Class::scale))
		.def("translate", &Class::translate)

		.def("setExtents", &Class::setExtents)

		.def("getGridSpacing", &Class::getGridSpacing)
		.def("setGridSpacing", &Class::setGridSpacing)

		.def("getOrigin", &Class::getOrigin)
		.def("setOrigin", &Class::setOrigin)
		.def("width", &Class::width)
		.def("height", &Class::height)

		.def("getDisplacement", &Class::getDisplacement, py::return_value_policy::reference)
		.def("calculateHeightField", &Class::calculateHeightField, py::return_value_policy::reference);
}

#include "Topology/QuadSet.h"
template <typename TDataType>
void declare_quad_set(py::module& m, std::string typestr) {
	using Class = dyno::QuadSet<TDataType>;
	using Parent = dyno::EdgeSet<TDataType>;
	typedef typename dyno::TopologyModule::Quad Quad;

	class QuadSetTrampoline : public Class
	{
	public:
		using Class::Class;

		void updateTopology() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::QuadSet<TDataType>,
				updateTopology
			);
		}

		void updateEdges() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::QuadSet<TDataType>,
				updateEdges
			);
		}
	};

	class QuadSetPublicist : public Class
	{
	public:
		using Class::updateTopology;
		using Class::updateEdges;
		using Class::updateQuads;
	};

	std::string pyclass_name = std::string("QuadSet") + typestr;
	py::class_<Class, Parent, QuadSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())

		.def("set_quads", py::overload_cast<std::vector<Quad>&>(&Class::setQuads))
		.def("set_quads", py::overload_cast<dyno::Array<Quad, DeviceType::GPU>&>(&Class::setQuads))

		.def("quadIndices", &Class::quadIndices, py::return_value_policy::reference)
		.def("vertex2Quad", &Class::vertex2Quad, py::return_value_policy::reference)
		.def("copyFrom", &Class::copyFrom)
		.def("isEmpty", &Class::isEmpty)
		.def("requestVertexNormals", &Class::requestVertexNormals, py::return_value_policy::reference)
		// protected
		.def("updateTopology", &QuadSetPublicist::updateTopology)
		.def("updateEdges", &QuadSetPublicist::updateEdges)
		.def("updateQuads", &QuadSetPublicist::updateQuads);
}

#include "Topology/HexahedronSet.h"
template <typename TDataType>
void declare_hexahedron_set(py::module& m, std::string typestr) {
	using Class = dyno::HexahedronSet<TDataType>;
	using Parent = dyno::QuadSet<TDataType>;
	typedef typename dyno::TopologyModule::Hexahedron Hexahedron;

	class HexahedronSetTrampoline : public Class
	{
	public:
		using Class::Class;

		void updateQuads() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::HexahedronSet<TDataType>,
				updateQuads
			);
		}
	};

	class HexahedronSetPublicist : public Class
	{
	public:
		using Class::updateQuads;
	};

	std::string pyclass_name = std::string("HexahedronSet") + typestr;
	py::class_<Class, Parent, HexahedronSetTrampoline,std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_hexahedrons", py::overload_cast<std::vector<Hexahedron>&>(&Class::setHexahedrons))
		.def("set_hexahedrons", py::overload_cast<dyno::Array<Hexahedron, DeviceType::GPU>&>(&Class::setHexahedrons))

		.def("getHexahedrons", &Class::getHexahedrons, py::return_value_policy::reference)
		.def("getQua2Hex", &Class::getQua2Hex, py::return_value_policy::reference)
		.def("getVer2Hex", &Class::getVer2Hex, py::return_value_policy::reference)
		.def("getVolume", &Class::getVolume)
		.def("copyFrom", &Class::copyFrom)
		// protected
		.def("updateQuads", &HexahedronSetPublicist::updateQuads);
}

#include "Topology/JointInfo.h"
inline void declare_joint_info(py::module& m) {
	using Class = dyno::JointInfo;
	using Parent = dyno::OBase;
	typedef int joint;
	std::string pyclass_name = std::string("JointInfo");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//.def(py::init<dyno::Array<dyno::Mat4f, DeviceType::GPU>&, dyno::Array<dyno::Mat4f, DeviceType::GPU>&, dyno::Array<dyno::Mat4f, DeviceType::GPU>&, std::vector<int>&, std::map<joint, std::vector<joint>>&, std::map<joint, dyno::Vec3f>&, std::map<joint, dyno::Vec3f>&, std::map<joint, dyno::Quat1f>&>())
		.def("setGltfJointInfo", &Class::setGltfJointInfo)
		.def("clear", &Class::clear)
		.def("SetJointInfo", &Class::SetJointInfo)
		.def("setJoint", &Class::setJoint)
		.def("isEmpty", &Class::isEmpty)
		.def("updateWorldMatrixByTransform", &Class::updateWorldMatrixByTransform)
		.def("setJointName", &Class::setJointName)
		.def("setLeftHandedCoordSystem", &Class::setLeftHandedCoordSystem)
		.def("setPose", &Class::setPose)
		.def("findJointIndexByName", &Class::findJointIndexByName)
		.def("getLocalMatrix", &Class::getLocalMatrix)
		.def_readwrite("mJointName", &Class::mJointName)

		.def_readwrite("mJointInverseBindMatrix", &Class::mJointInverseBindMatrix)
		.def_readwrite("mJointLocalMatrix", &Class::mJointLocalMatrix)
		.def_readwrite("mJointWorldMatrix", &Class::mJointWorldMatrix)

		.def_readwrite("currentPose", &Class::currentPose)

		.def_readwrite("mBindPoseTranslation", &Class::mBindPoseTranslation)
		.def_readwrite("mBindPoseScale", &Class::mBindPoseScale)
		.def_readwrite("mBindPoseRotation", &Class::mBindPoseRotation)
		.def_readwrite("mBindPoseRotator", &Class::mBindPoseRotator)

		.def_readwrite("mBindPosePreRotator", &Class::mBindPosePreRotator)

		.def_readwrite("mAllJoints", &Class::mAllJoints)
		.def_readwrite("mJointDir", &Class::mJointDir)

		.def_readwrite("mMaxJointID", &Class::mMaxJointID);
}

inline void declare_joint_animation_info(py::module& m) {
	using Class = dyno::JointAnimationInfo;
	using Parent = dyno::OBase;
	typedef int joint;
	std::string pyclass_name = std::string("JointAnimationInfo");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("setGLTFAnimationData", &Class::setGLTFAnimationData)
		.def("clear", &Class::clear)
		.def("isGltfAnimation", &Class::isGltfAnimation)
		.def("resizeJointsData", &Class::resizeJointsData)
		.def("getJointsTranslation", &Class::getJointsTranslation)
		.def("getJointsRotation", &Class::getJointsRotation)
		.def("getJointsScale", &Class::getJointsScale)
		.def("getTotalTime", &Class::getTotalTime)
		.def("findMaxSmallerIndex", &Class::findMaxSmallerIndex)

		//.def("lerp", py::overload_cast<std::vector<Coord>, std::vector<dyno::TopologyModule::Triangle>&>(&Class::lerp))
		//override
		.def("normalize", &Class::normalize)
		.def("slerp", &Class::slerp)
		.def("nlerp", &Class::nlerp)
		.def("getJointDir", &Class::getJointDir)
		.def("setLoop", &Class::setLoop)
		.def("getPose", &Class::getPose)
		.def("getCurrentAnimationTime", &Class::getCurrentAnimationTime)
		.def("getBlendInTime", &Class::getBlendInTime)
		.def("getBlendOutTime", &Class::getBlendOutTime)
		.def("getPlayRate", &Class::getPlayRate)

		.def_readwrite("mSkeleton", &Class::mSkeleton);
}

#include "Topology/LevelSet.h"
template <typename TDataType>
void declare_level_set(py::module& m, std::string typestr) {
	using Class = dyno::LevelSet<TDataType>;
	using Parent = dyno::TopologyModule;
	std::string pyclass_name = std::string("SignedDistanceField") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getSDF", &Class::getSDF, py::return_value_policy::reference)
		.def("setSDF", &Class::setSDF);
}

#include "Topology/LinearBVH.h"
template <typename TDataType>
void declare_linear_bvh(py::module& m, std::string typestr) {
	using Class = dyno::LinearBVH<TDataType>;
	std::string pyclass_name = std::string("LinearBVH") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("construct", &Class::construct)

		.def("requestIntersectionNumber", &Class::requestIntersectionNumber)
		.def("requestIntersectionIds", &Class::requestIntersectionIds)

		.def("getRoot", &Class::getRoot)

		.def("getAABB", &Class::getAABB)
		.def("getObjectIdx", &Class::getObjectIdx)

		.def("getSortedAABBs", &Class::getSortedAABBs, py::return_value_policy::reference)
		.def("release", &Class::release);
}

#include "Topology/PolygonSet.h"
template <typename TDataType>
void declare_polygon_set(py::module& m, std::string typestr) {
	using Class = dyno::PolygonSet<TDataType>;
	using Parent = dyno::EdgeSet<TDataType>;

	class PolygonSetTrampoline : public Class
	{
	public:
		using Class::Class;

		void updateTopology() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PolygonSet<TDataType>,
				updateTopology
			);
		}

		void updateEdges() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PolygonSet<TDataType>,
				updateEdges
			);
		}
	};

	class PolygonSetPublicist : public Class
	{
	public:
		using Class::updateTopology;
		using Class::updateEdges;
	};

	std::string pyclass_name = std::string("PolygonSet") + typestr;
	py::class_<Class, Parent, PolygonSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("setPolygons", py::overload_cast<const dyno::ArrayList<dyno::uint, DeviceType::CPU>&>(&Class::setPolygons))
		.def("setPolygons", py::overload_cast<const dyno::ArrayList<dyno::uint, DeviceType::GPU>&>(&Class::setPolygons))

		.def("polygonIndices", &Class::polygonIndices, py::return_value_policy::reference)
		.def("vertex2Polygon", &Class::vertex2Polygon, py::return_value_policy::reference)
		.def("polygon2Edge", &Class::polygon2Edge, py::return_value_policy::reference)
		.def("edge2Polygon", &Class::edge2Polygon, py::return_value_policy::reference)
		.def("copyFrom", &Class::copyFrom)
		.def("isEmpty", &Class::isEmpty)
		.def("extractEdgeSet", &Class::extractEdgeSet)
		.def("extractTriangleSet", &Class::extractTriangleSet)
		.def("extractQuadSet", &Class::extractQuadSet)
		.def("turnIntoTriangleSet", &Class::turnIntoTriangleSet)
		.def("triangleSetToPolygonSet", &Class::triangleSetToPolygonSet)
		// protected
		.def("updateTopology", &PolygonSetPublicist::updateTopology)
		.def("updateEdges", &PolygonSetPublicist::updateEdges);

}

#include "Topology/SimplexSet.h"
template <typename TDataType>
void declare_simplex_set(py::module& m, std::string typestr) {
	using Class = dyno::SimplexSet<TDataType>;
	using Parent = dyno::PointSet<TDataType>;
	typedef typename dyno::TopologyModule::Edge Edge;
	typedef typename dyno::TopologyModule::Triangle Triangle;
	typedef typename dyno::TopologyModule::Tetrahedron Tetrahedron;

	class SimplexSetTrampoline : public Class
	{
	public:
		using Class::Class;

		void updateTopology() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SimplexSet<TDataType>,
				updateTopology
			);
		}
	};

	class SimplexSetPublicist : public Class
	{
	public:
		using Class::updateTopology;
	};

	std::string pyclass_name = std::string("SimplexSet") + typestr;
	py::class_<Class, Parent, SimplexSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("copyFrom", &Class::copyFrom)
		.def("isEmpty", &Class::isEmpty)
		.def("setEdgeIndex", py::overload_cast<const dyno::Array<Edge, DeviceType::GPU>&>(&Class::setEdgeIndex))
		.def("setEdgeIndex", py::overload_cast<const dyno::Array<Edge, DeviceType::CPU>&>(&Class::setEdgeIndex))
		.def("setTriangleIndex", py::overload_cast<const dyno::Array<Triangle, DeviceType::GPU>&>(&Class::setTriangleIndex))
		.def("setTriangleIndex", py::overload_cast<const dyno::Array<Triangle, DeviceType::CPU>&>(&Class::setTriangleIndex))
		.def("setTetrahedronIndex", py::overload_cast<const dyno::Array<Tetrahedron, DeviceType::GPU>&>(&Class::setTetrahedronIndex))
		.def("setTetrahedronIndex", py::overload_cast<const dyno::Array<Tetrahedron, DeviceType::CPU>&>(&Class::setTetrahedronIndex))
		.def("extractSimplex1D", &Class::extractSimplex1D)
		.def("extractSimplex2D", &Class::extractSimplex2D)
		.def("extractSimplex3D", &Class::extractSimplex3D)
		.def("extractPointSet", &Class::extractPointSet)
		.def("extractEdgeSet", &Class::extractEdgeSet)
		.def("extractTriangleSet", &Class::extractTriangleSet)
		// protected
		.def("updateTopology", &SimplexSetPublicist::updateTopology);
}

#include "Topology/SparseGridHash.h"
template <typename TDataType>
void declare_sparse_grid_hash(py::module& m, std::string typestr) {
	using Class = dyno::SparseGridHash<TDataType>;
	std::string pyclass_name = std::string("SparseGridHash") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("setSpace", &Class::setSpace)
		.def("construct", &Class::construct);
}

#include "Topology/SparseOctree.h"
template <typename TDataType>
void declare_sparse_octree(py::module& m, std::string typestr) {
	using Class = dyno::SparseOctree<TDataType>;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	std::string pyclass_name = std::string("SparseOctree") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("release", &Class::release)
		.def("setSpace", &Class::setSpace)

		.def("construct", py::overload_cast<const dyno::Array<Coord, DeviceType::GPU>&, Real>(&Class::construct))
		.def("construct", py::overload_cast<const dyno::Array<dyno::AABB, DeviceType::GPU>&>(&Class::construct))
		.def("construct", py::overload_cast<const dyno::Array<dyno::OctreeNode, DeviceType::GPU>&>(&Class::construct))

		.def("getLevelMax", &Class::getLevelMax)

		.def("queryNode", &Class::queryNode)

		.def("requestLevelNumber", &Class::requestLevelNumber)

		//.def("requestIntersectionNumber", &Class::requestIntersectionNumber)
		//.def("reqeustIntersectionIds", &Class::reqeustIntersectionIds)

		.def("requestIntersectionNumberFromLevel", py::overload_cast<const dyno::AABB, int>(&Class::requestIntersectionNumberFromLevel))
		.def("requestIntersectionNumberFromLevel", py::overload_cast<const dyno::AABB, dyno::AABB*, int>(&Class::requestIntersectionNumberFromLevel))

		.def("reqeustIntersectionIdsFromLevel", py::overload_cast<int*, const dyno::AABB, int>(&Class::reqeustIntersectionIdsFromLevel))
		.def("reqeustIntersectionIdsFromLevel", py::overload_cast<int*, const dyno::AABB, dyno::AABB*, int>(&Class::reqeustIntersectionIdsFromLevel))

		.def("requestIntersectionNumberFromBottom", py::overload_cast<const dyno::AABB>(&Class::requestIntersectionNumberFromBottom))
		.def("requestIntersectionNumberFromBottom", py::overload_cast<const dyno::AABB, dyno::AABB*>(&Class::requestIntersectionNumberFromBottom))

		.def("reqeustIntersectionIdsFromBottom", py::overload_cast<int*, const dyno::AABB>(&Class::reqeustIntersectionIdsFromBottom))
		.def("reqeustIntersectionIdsFromBottom", py::overload_cast<int*, const dyno::AABB, dyno::AABB*>(&Class::reqeustIntersectionIdsFromBottom))

		.def("printAllNodes", &Class::printAllNodes)
		.def("printPostOrderedTree", &Class::printPostOrderedTree);
}

#include "Topology/StructuredPointSet.h"
template <typename TDataType>
void declare_structured_point_set(py::module& m, std::string typestr) {
	using Class = dyno::StructuredPointSet<TDataType>;
	using Parent = dyno::PointSet<TDataType>;
	std::string pyclass_name = std::string("StructuredPointSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

#include "Topology/TriangleSet.h"
template <typename TDataType>
void declare_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::TriangleSet<TDataType>;
	using Parent = dyno::EdgeSet<TDataType>;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename dyno::TopologyModule::Triangle Triangle;

	class TriangleSetTrampoline : public Class
	{
	public:
		using Class::Class;

		void updateTopology() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TriangleSet<TDataType>,
				updateTopology
			);
		}

		void updateEdges() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TriangleSet<TDataType>,
				updateEdges
			);
		}
	};

	class TriangleSetPublicist : public Class
	{
	public:
		using Class::updateTopology;
		using Class::updateEdges;
		using Class::updateTriangles;
	};

	std::string pyclass_name = std::string("TriangleSet") + typestr;
	py::class_<Class, Parent, TriangleSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("setTriangles", py::overload_cast<std::vector<Triangle>&>(&Class::setTriangles))
		.def("setTriangles", py::overload_cast<dyno::Array<Triangle, DeviceType::GPU>&>(&Class::setTriangles))

		.def("triangleIndices", &Class::triangleIndices)
		.def("vertex2Triangle", &Class::vertex2Triangle)

		.def("triangle2Edge", &Class::triangle2Edge)
		.def("edge2Triangle", &Class::edge2Triangle)

		.def("loadObjFile", &Class::loadObjFile)
		.def("copyFrom", &Class::copyFrom)
		.def("merge", &Class::merge)
		.def("isEmpty", &Class::isEmpty)
		.def("clear", &Class::clear)

		.def("requestTriangle2Triangle", &Class::requestTriangle2Triangle)
		.def("requestEdgeNormals", &Class::requestEdgeNormals)
		.def("requestVertexNormals", &Class::requestVertexNormals)
		// protected
		.def("updateTopology", &TriangleSetPublicist::updateTopology)
		.def("updateEdges", &TriangleSetPublicist::updateEdges)
		.def("updateTriangles", &TriangleSetPublicist::updateTriangles);
}

#include "Topology/TetrahedronSet.h"
template <typename TDataType>
void declare_tetrahedron_set(py::module& m, std::string typestr) {
	using Class = dyno::TetrahedronSet<TDataType>;
	using Parent = dyno::TriangleSet<TDataType>;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename dyno::TopologyModule::Triangle Triangle;
	typedef typename dyno::TopologyModule::Tetrahedron Tetrahedron;

	class TetrahedronSetTrampoline : public Class
	{
	public:
		using Class::Class;

		void updateTopology() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TetrahedronSet<TDataType>,
				updateTopology
			);
		}

		void updateTriangles() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TetrahedronSet<TDataType>,
				updateTriangles
			);
		}
	};

	class TetrahedronSetPublicist : public Class
	{
	public:
		using Class::updateTopology;
		using Class::updateTriangles;
		using Class::updateTetrahedrons;
	};

	std::string pyclass_name = std::string("TetrahedronSet") + typestr;
	py::class_<Class, Parent, TetrahedronSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("loadTetFile", &Class::loadTetFile)

		.def("setTetrahedrons", py::overload_cast<std::vector<Tetrahedron>&>(&Class::setTetrahedrons))
		.def("setTetrahedrons", py::overload_cast<dyno::Array<Tetrahedron, DeviceType::GPU>&>(&Class::setTetrahedrons))

		.def("tetrahedronIndices", &Class::tetrahedronIndices, py::return_value_policy::reference)
		.def("vertex2Tetrahedron", &Class::vertex2Tetrahedron, py::return_value_policy::reference)
		.def("triangle2Tetrahedron", &Class::triangle2Tetrahedron, py::return_value_policy::reference)
		.def("tetrahedron2Triangle", &Class::tetrahedron2Triangle, py::return_value_policy::reference)
		.def("copyFrom", &Class::copyFrom)
		.def("isEmpty", &Class::isEmpty)

		.def("requestPointNeighbors", &Class::requestPointNeighbors, py::return_value_policy::reference)
		.def("requestSurfaceMeshIds", &Class::requestSurfaceMeshIds, py::return_value_policy::reference)
		.def("extractSurfaceMesh", &Class::extractSurfaceMesh, py::return_value_policy::reference)
		.def("calculateVolume", &Class::calculateVolume, py::return_value_policy::reference)
		// protected
		.def("updateTopology", &TetrahedronSetPublicist::updateTopology)
		.def("updateTriangles", &TetrahedronSetPublicist::updateTriangles)
		.def("updateTetrahedrons", &TetrahedronSetPublicist::updateTetrahedrons);
}

#include "Topology/UniformGrid.h"
template <typename TDataType>
void declare_uniform_grid3D(py::module& m, std::string typestr) {
	using Class = dyno::UniformGrid3D<TDataType>;
	std::string pyclass_name = std::string("UniformGrid3D") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

#include "Topology/UnstructuredPointSet.h"
template <typename TDataType>
void declare_unstructured_point_set(py::module& m, std::string typestr) {
	using Class = dyno::UnstructuredPointSet<TDataType>;
	using Parent = dyno::PointSet<TDataType>;
	std::string pyclass_name = std::string("UnstructuredPointSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("copyFrom", &Class::copyFrom)
		.def("getPointNeighbors", &Class::getPointNeighbors, py::return_value_policy::reference)
		.def("clear", &Class::clear);
}

void declare_texture_mesh(py::module& m);

void declare_attribute(py::module& m);

void pybind_topology(py::module& m);