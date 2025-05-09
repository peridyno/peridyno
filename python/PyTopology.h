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
		.def("in_discrete_elements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("out_aabb", &Class::outAABB, py::return_value_policy::reference);
}

#include "Collision/CollisionDetectionBroadPhase.h"
template<typename TDataType>
void declare_collision_detection_broad_phase(py::module& m, std::string typestr) {
	using Class = dyno::CollisionDetectionBroadPhase<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("CollisionDetectionBroadPhase") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>CDBP(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	CDBP.def(py::init<>())
		.def("var_acceleration_structure", &Class::varAccelerationStructure, py::return_value_policy::reference)
		.def("var_grid_size_limit", &Class::varGridSizeLimit, py::return_value_policy::reference)
		.def("var_self_collision", &Class::varSelfCollision, py::return_value_policy::reference)
		.def("in_source", &Class::inSource, py::return_value_policy::reference)
		.def("in_target", &Class::inTarget, py::return_value_policy::reference)
		.def("out_contact_list", &Class::outContactList, py::return_value_policy::reference);

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
	std::string pyclass_name = std::string("CollistionDetectionBoundingBox") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_upper_bound", &Class::varUpperBound, py::return_value_policy::reference)
		.def("var_lower_bound", &Class::varLowerBound, py::return_value_policy::reference)
		.def("in_discrete_elements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("out_contacts", &Class::outContacts, py::return_value_policy::reference);
}

#include "Collision/CollistionDetectionTriangleSet.h"
template<typename TDataType>
void declare_collistion_detection_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::CollistionDetectionTriangleSet<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("CollistionDetectionTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_discrete_elements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("out_contacts", &Class::outContacts, py::return_value_policy::reference);
}

#include "Collision/NeighborElementQuery.h"
template<typename TDataType>
void declare_neighbor_element_query(py::module& m, std::string typestr) {
	using Class = dyno::NeighborElementQuery<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("NeighborElementQuery") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_self_collision", &Class::varSelfCollision, py::return_value_policy::reference)
		.def("var_d_head", &Class::varDHead, py::return_value_policy::reference)

		.def("var_grid_size_limit", &Class::varGridSizeLimit, py::return_value_policy::reference)
		.def("in_discrete_elements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("in_collision_mask", &Class::inCollisionMask, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("out_contacts", &Class::outContacts, py::return_value_policy::reference);
}

#include "Collision/NeighborPointQuery.h"
template<typename TDataType>
void declare_neighbor_point_query(py::module& m, std::string typestr) {
	using Class = dyno::NeighborPointQuery<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("NeighborPointQuery") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>NPQ(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	NPQ.def(py::init<>())
		.def("var_spatial", &Class::varSpatial, py::return_value_policy::reference)
		.def("var_size_limit", &Class::varSizeLimit, py::return_value_policy::reference)
		.def("in_radius", &Class::inRadius, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_other", &Class::inOther, py::return_value_policy::reference)
		.def("out_neighbor_ids", &Class::outNeighborIds, py::return_value_policy::reference);

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
	std::string pyclass_name = std::string("NeighborTriangleQuery") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>NTQ(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	NTQ.def(py::init<>())
		.def("var_spatial", &Class::varSpatial, py::return_value_policy::reference)
		.def("in_radius", &Class::inRadius, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("out_neighbor_ids", &Class::outNeighborIds, py::return_value_policy::reference);

	py::enum_<typename Class::Spatial>(NTQ, "Spatial")
		.value("BVH", Class::Spatial::BVH)
		.value("OCTREE", Class::Spatial::OCTREE);
}

#include "Mapping/AnchorPointToPointSet.h"
template <typename TDataType>
void declare_anchor_point_to_point_set(py::module& m, std::string typestr) {
	using Class = dyno::AnchorPointToPointSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("AnchorPointToPointSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_discrete_elements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("in_center", &Class::inCenter, py::return_value_policy::reference)
		.def("in_rotation_matrix", &Class::inRotationMatrix, py::return_value_policy::reference)
		.def("out_point_set", &Class::outPointSet, py::return_value_policy::reference);
}

#include "Mapping/BoundingBoxToEdgeSet.h"
template <typename TDataType>
void declare_bounding_box_to_edge_set(py::module& m, std::string typestr) {
	using Class = dyno::BoundingBoxToEdgeSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("BoundingBoxToEdgeSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_aabb", &Class::inAABB, py::return_value_policy::reference)
		.def("out_edge_set", &Class::outEdgeSet, py::return_value_policy::reference);
}

#include "Mapping/ContactsToEdgeSet.h"
template<typename TDataType>
void declare_contacts_to_edge_set(py::module& m, std::string typestr) {
	using Class = dyno::ContactsToEdgeSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("ContactsToEdgeSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_scale", &Class::varScale, py::return_value_policy::reference)
		.def("in_contacts", &Class::inContacts, py::return_value_policy::reference)
		.def("out_edge_set", &Class::outEdgeSet, py::return_value_policy::reference);
}

#include "Mapping/ContactsToPointSet.h"
template<typename TDataType>
void declare_contacts_to_point_set(py::module& m, std::string typestr) {
	using Class = dyno::ContactsToPointSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("ContactsToPointSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_contacts", &Class::inContacts, py::return_value_policy::reference)
		.def("out_point_set", &Class::outPointSet, py::return_value_policy::reference);
}

#include "Mapping/DiscreteElementsToTriangleSet.h"
template <typename TDataType>
void declare_discrete_elements_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::DiscreteElementsToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("DiscreteElementsToTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_discrete_elements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference);
}

#include "Mapping/Extract.h"
template <typename TDataType>
void declare_extract_edge_set_from_polygon_set(py::module& m, std::string typestr) {
	using Class = dyno::ExtractEdgeSetFromPolygonSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("ExtractEdgeSetFromPolygonSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("in_polygon_set", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("out_edge_set", &Class::outEdgeSet, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_extract_triangle_set_from_polygon_set(py::module& m, std::string typestr) {
	using Class = dyno::ExtractTriangleSetFromPolygonSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("ExtractTriangleSetFromPolygonSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("in_polygon_set", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_extract_qaud_set_from_polygon_set(py::module& m, std::string typestr) {
	using Class = dyno::ExtractQaudSetFromPolygonSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("ExtractQaudSetFromPolygonSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("in_polygon_set", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("out_quad_set", &Class::outQuadSet, py::return_value_policy::reference);
}

#include "Mapping/FrameToPointSet.h"
template <typename TDataType>
void declare_frame_to_point_set(py::module& m, std::string typestr) {
	using Class = dyno::FrameToPointSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("FrameToPointSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("initialize", &Class::initialize)
		.def("apply_transform", &Class::applyTransform)
		.def("apply", &Class::apply);
}

#include "Mapping/HeightFieldToTriangleSet.h"
template <typename TDataType>
void declare_height_field_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::HeightFieldToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("HeightFieldToTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_height_field", &Class::inHeightField, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference)
		.def("var_scale", &Class::varScale, py::return_value_policy::reference)
		.def("var_translation", &Class::varTranslation, py::return_value_policy::reference);
}

#include "Mapping/MergeSimplexSet.h"
template <typename TDataType>
void declare_merge_simplex_set(py::module& m, std::string typestr) {
	using Class = dyno::MergeSimplexSet<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("MergeSimplexSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_edge_Set", &Class::inEdgeSet, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("in_tetrahedron_set", &Class::inTetrahedronSet, py::return_value_policy::reference)
		.def("out_simplex_set", &Class::outSimplexSet, py::return_value_policy::reference);
}

#include "Mapping/MergeTriangleSet.h"
template <typename TDataType>
void declare_merge_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::MergeTriangleSet<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("MergeTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("in_first", &Class::inFirst, py::return_value_policy::reference)
		.def("in_second", &Class::inSecond, py::return_value_policy::reference);
}

#include "Mapping/PointSetToPointSet.h"
template <typename TDataType>
void declare_point_set_to_point_set(py::module& m, std::string typestr) {
	using Class = dyno::PointSetToPointSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("PointSetToPointSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<std::shared_ptr<dyno::PointSet<TDataType>>, std::shared_ptr<dyno::PointSet<TDataType>> >())
		.def("set_searching_radius", &Class::setSearchingRadius)
		.def("set_from", &Class::setFrom)
		.def("set_to", &Class::setTo)
		.def("apply", &Class::apply)
		.def("match", &Class::match);
}

#include "Mapping/PointSetToTriangleSet.h"
template <typename TDataType>
void declare_point_set_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::PointSetToTriangleSet<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("PointSetToTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_point_set", &Class::inPointSet, py::return_value_policy::reference)
		.def("in_initial_shape", &Class::inInitialShape, py::return_value_policy::reference)
		.def("out_shape", &Class::outShape, py::return_value_policy::reference);
}

#include "Mapping/QuadSetToTriangleSet.h"
template <typename TDataType>
void declare_quad_set_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::QuadSetToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("QuadSetToTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_quad_set", &Class::inQuadSet, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference);
}

#include "Mapping/SplitSimplexSet.h"
template <typename TDataType>
void declare_split_simplex_set(py::module& m, std::string typestr) {
	using Class = dyno::SplitSimplexSet<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("SplitSimplexSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_simplex_set", &Class::inSimplexSet, py::return_value_policy::reference)
		.def("out_edge_set", &Class::outEdgeSet, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference)
		.def("out_tetrahedron_set", &Class::outTetrahedronSet, py::return_value_policy::reference);
}

#include "Mapping/TetrahedronSetToPointSet.h"
template <typename TDataType>
void declare_tetrahedron_set_to_point_set(py::module& m, std::string typestr) {
	using Class = dyno::TetrahedronSetToPointSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("TetrahedronSetToPointSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_searching_radius", &Class::setSearchingRadius)
		.def("set_from", &Class::setFrom)
		.def("set_to", &Class::setTo)
		.def("match", &Class::match);
}

#include "Mapping/TextureMeshToTriangleSet.h"
template <typename TDataType>
void declare_texture_mesh_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::TextureMeshToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("TextureMeshToTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_texture_mesh", &Class::inTextureMesh, py::return_value_policy::reference)
		.def("in_transform", &Class::inTransform, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_texture_mesh_to_triangle_set_node(py::module& m, std::string typestr) {
	using Class = dyno::TextureMeshToTriangleSetNode<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("TextureMeshToTriangleSetNode") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("in_texture_mesh", &Class::inTextureMesh, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference);
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
		.def("in_scalar_array", &Class::inScalarArray, py::return_value_policy::reference)
		.def("out_scalar", &Class::outScalar, py::return_value_policy::reference);
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
		.def("in_scalar_array", &Class::inScalarArray, py::return_value_policy::reference)
		.def("out_scalar", &Class::outScalar, py::return_value_policy::reference);
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
		.def("in_vec", &Class::inVec, py::return_value_policy::reference)
		.def("out_norm", &Class::outNorm, py::return_value_policy::reference);
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
		.def("fbx_time_to_seconds", &Class::fbxTimeToSeconds)
		.def("seconds_to_fbx_time", &Class::secondsToFbxTime)
		.def("set_init_val", &Class::setInitVal)
		.def("get_curve_value_along", &Class::getCurveValueAlong)
		.def("get_curve_value_all", &Class::getCurveValueAll)
		.def("get_curve_value_cycle", &Class::getCurveValueCycle)

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
		.def("set_anchor_point", &Class::setAnchorPoint)
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
		.def("set_anchor_point", &Class::setAnchorPoint)
		.def("set_axis", &Class::setAxis)
		.def("set_moter", &Class::setMoter)
		.def("set_range", &Class::setRange)

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
		.def("set_anchor_point", &Class::setAnchorPoint)
		.def("set_axis", &Class::setAxis)
		.def("set_moter", &Class::setMoter)
		.def("set_range", &Class::setRange)

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
		.def("set_anchor_point", &Class::setAnchorPoint)
		.def("set_anchor_angle", &Class::setAnchorAngle)

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
		.def("set_anchor_point", &Class::setAnchorPoint)

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
		.def("set_distance_joint", &Class::setDistanceJoint)

		.def_readwrite("r1", &Class::r1)
		.def_readwrite("r2", &Class::r2)
		.def_readwrite("distance", &Class::distance);
}

template <typename TDataType>
void declare_discrete_elements(py::module& m, std::string typestr) {
	using Class = dyno::DiscreteElements<TDataType>;
	using Parent = dyno::TopologyModule;
	std::string pyclass_name = std::string("DiscreteElements") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("scale", &Class::scale)

		.def("total_size", &Class::totalSize)

		.def("total_joint_size", &Class::totalJointSize)

		.def("sphere_index", &Class::sphereIndex)
		.def("box_index", &Class::boxIndex)
		.def("capsule_index", &Class::capsuleIndex)
		.def("tet_index", &Class::tetIndex)
		.def("triangle_index", &Class::triangleIndex)

		.def("calculate_element_offset", &Class::calculateElementOffset)

		.def("set_spheres", &Class::setSpheres)
		.def("set_boxes", &Class::setBoxes)
		.def("set_tets", &Class::setTets)
		.def("set_capsules", &Class::setCapsules)
		.def("set_triangles", &Class::setTriangles)
		.def("set_tet_sdf", &Class::setTetSDF)

		.def("spheres_in_local", &Class::spheresInLocal, py::return_value_policy::reference)
		.def("boxes_in_local", &Class::boxesInLocal, py::return_value_policy::reference)
		.def("tets_in_local", &Class::tetsInLocal, py::return_value_policy::reference)
		.def("capsules_in_local", &Class::capsulesInLocal, py::return_value_policy::reference)
		.def("triangles_in_local", &Class::trianglesInLocal, py::return_value_policy::reference)

		.def("spheres_in_global", &Class::spheresInGlobal, py::return_value_policy::reference)
		.def("boxes_in_global", &Class::boxesInGlobal, py::return_value_policy::reference)
		.def("tets_in_global", &Class::tetsInGlobal, py::return_value_policy::reference)
		.def("capsules_in_global", &Class::capsulesInGlobal, py::return_value_policy::reference)
		.def("triangles_in_global", &Class::trianglesInGlobal, py::return_value_policy::reference)

		.def("shape_2_rigid_body_mapping", &Class::shape2RigidBodyMapping, py::return_value_policy::reference)

		.def("position", &Class::position, py::return_value_policy::reference)
		.def("rotation", &Class::rotation, py::return_value_policy::reference)

		.def("set_position", &Class::setPosition)
		.def("set_rotation", &Class::setRotation)

		.def("ball_and_socket_joints", &Class::ballAndSocketJoints, py::return_value_policy::reference)
		.def("slider_joints", &Class::sliderJoints, py::return_value_policy::reference)
		.def("hinge_joints", &Class::hingeJoints, py::return_value_policy::reference)
		.def("fixed_joints", &Class::fixedJoints, py::return_value_policy::reference)
		.def("point_joints", &Class::pointJoints, py::return_value_policy::reference)
		.def("distance_joints", &Class::distanceJoints, py::return_value_policy::reference)

		.def("set_tet_body_id", &Class::setTetBodyId)
		.def("set_tet_element_id", &Class::setTetElementId)

		.def("get_tet_sdf", &Class::getTetSDF, py::return_value_policy::reference)
		.def("get_tet_body_mapping", &Class::getTetBodyMapping, py::return_value_policy::reference)
		.def("get_tet_element_mapping", &Class::getTetElementMapping, py::return_value_policy::reference)

		.def("copy_from", &Class::copyFrom)
		.def("merge", &Class::merge)
		.def("request_discrete_elements_in_global", &Class::requestDiscreteElementsInGlobal)

		.def("request_box_in_global", &Class::requestBoxInGlobal, py::return_value_policy::reference)
		.def("request_sphere_in_global", &Class::requestSphereInGlobal, py::return_value_policy::reference)
		.def("request_tet_in_global", &Class::requestTetInGlobal, py::return_value_policy::reference)
		.def("request_capsule_in_global", &Class::requestCapsuleInGlobal, py::return_value_policy::reference)
		.def("request_triangle_in_global", &Class::requestTriangleInGlobal, py::return_value_policy::reference);
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
		.def("get_distance", &Class::getDistance)
		.def("nx", &Class::nx)
		.def("ny", &Class::ny)
		.def("nz", &Class::nz)
		.def("load_sdf", &Class::loadSDF)
		.def("load_box", &Class::loadBox)
		.def("load_cylinder", &Class::loadCylinder)
		.def("load_sphere", &Class::loadSphere)
		.def("set_space", &Class::setSpace)
		.def("lower_bound", &Class::lowerBound)
		.def("upper_bound", &Class::upperBound)
		.def("assign", &Class::assign)
		.def("get_m_distance", &Class::distances, py::return_value_policy::reference)
		.def("set_distance", &Class::setDistance)
		.def("get_h", &Class::getGridSpacing)
		.def("invert_sdf", &Class::invertSDF);
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
		.def("copy_from", &Class::copyFrom)
		.def("set_points", py::overload_cast<const std::vector<Coord>&>(&Class::setPoints))
		.def("set_points", py::overload_cast<const dyno::Array<Coord, DeviceType::GPU>&>(&Class::setPoints))
		.def("set_size", &Class::setSize)

		.def("get_point_size", &Class::getPointSize)

		.def("request_bounding_box", &Class::requestBoundingBox)
		.def("scale", py::overload_cast<const Real>(&Class::scale))
		.def("scale", py::overload_cast<const Coord>(&Class::scale))
		.def("translate", &Class::translate)

		.def("rotate", py::overload_cast<const Coord>(&Class::rotate))
		.def("rotate", py::overload_cast<const dyno::Quat<Real>>(&Class::rotate))

		.def("load_obj_file", &Class::loadObjFile)
		.def("is_empty", &Class::isEmpty)
		.def("clear", &Class::clear)
		.def("get_points", &Class::getPoints);
}

#include "Topology/EdgeSet.h"
template <typename TDataType>
void declare_edge_set(py::module& m, std::string typestr) {
	using Class = dyno::EdgeSet<TDataType>;
	using Parent = dyno::PointSet<TDataType>;
	typedef typename dyno::TopologyModule::Edge Edge;
	std::string pyclass_name = std::string("EdgeSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_edges", py::overload_cast<std::vector<Edge>&>(&Class::setEdges))
		.def("set_edges", py::overload_cast<dyno::Array<Edge, DeviceType::GPU>&>(&Class::setEdges))
		.def("request_point_neighbors", &Class::requestPointNeighbors)
		.def("get_edges", &Class::getEdges)
		.def("vertex_2_edge", &Class::vertex2Edge)
		.def("copy_from", &Class::copyFrom)
		.def("is_empty", &Class::isEmpty)
		.def("clear", &Class::clear)
		.def("load_smesh_file", &Class::loadSmeshFile);
}

#include "Topology/Frame.h"
template <typename TDataType>
void declare_frame(py::module& m, std::string typestr) {
	using Class = dyno::Frame<TDataType>;
	using Parent = dyno::TopologyModule;
	std::string pyclass_name = std::string("Frame") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("copy_from", &Class::copyFrom)
		.def("set_center", &Class::setCenter)
		.def("get_center", &Class::getCenter)
		.def("set_orientation", &Class::setOrientation)
		.def("get_orientation", &Class::getOrientation);
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
		.def("set_space", &Class::setSpace)
		.def("construct", &Class::construct)
		.def("clear", &Class::clear)
		.def("release", &Class::release)
		.def("get_index", py::overload_cast<int, int, int>(&Class::getIndex))
		.def("get_index", py::overload_cast<Coord>(&Class::getIndex))
		.def("get_index3", &Class::getIndex3)
		.def("get_counter", &Class::getCounter)
		.def("get_particle_id", &Class::getParticleId)

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
		.def("set_uni_grid", &Class::setUniGrid)
		.def("set_nijk", &Class::setNijk)
		.def("set_origin", &Class::setOrigin)
		.def("set_dx", &Class::setDx)

		.def("get_ni", &Class::getNi)
		.def("get_nj", &Class::getNj)
		.def("get_nk", &Class::getNk)
		.def("get_origin", &Class::getOrigin)
		.def("get_dx", &Class::getDx);
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
		.def("copy_from", &Class::copyFrom)

		.def("scale", py::overload_cast<Real>(&Class::scale))
		.def("scale", py::overload_cast<Coord>(&Class::scale))
		.def("translate", &Class::translate)

		.def("set_extents", &Class::setExtents)

		.def("get_grid_spacing", &Class::getGridSpacing)
		.def("set_grid_spacing", &Class::setGridSpacing)

		.def("get_origin", &Class::getOrigin)
		.def("set_origin", &Class::setOrigin)
		.def("width", &Class::width)
		.def("height", &Class::height)

		.def("get_displacement", &Class::getDisplacement, py::return_value_policy::reference)
		.def("calculate_height_field", &Class::calculateHeightField, py::return_value_policy::reference);
}

#include "Topology/QuadSet.h"
template <typename TDataType>
void declare_quad_set(py::module& m, std::string typestr) {
	using Class = dyno::QuadSet<TDataType>;
	using Parent = dyno::EdgeSet<TDataType>;
	typedef typename dyno::TopologyModule::Quad Quad;
	std::string pyclass_name = std::string("QuadSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_quads", &Class::getQuads, py::return_value_policy::reference)

		.def("set_quads", py::overload_cast<std::vector<Quad>&>(&Class::setQuads))
		.def("set_quads", py::overload_cast<dyno::Array<Quad, DeviceType::GPU>&>(&Class::setQuads))

		.def("get_vertex2quads", &Class::getVertex2Quads, py::return_value_policy::reference)
		.def("copy_from", &Class::copyFrom)
		.def("is_empty", &Class::isEmpty)
		.def("out_vertex_normal", &Class::outVertexNormal, py::return_value_policy::reference);
}

#include "Topology/HexahedronSet.h"
template <typename TDataType>
void declare_hexahedron_set(py::module& m, std::string typestr) {
	using Class = dyno::HexahedronSet<TDataType>;
	using Parent = dyno::QuadSet<TDataType>;
	typedef typename dyno::TopologyModule::Hexahedron Hexahedron;
	std::string pyclass_name = std::string("HexahedronSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_hexahedrons", py::overload_cast<std::vector<Hexahedron>&>(&Class::setHexahedrons))
		.def("set_hexahedrons", py::overload_cast<dyno::Array<Hexahedron, DeviceType::GPU>&>(&Class::setHexahedrons))

		.def("get_hexahedrons", &Class::getHexahedrons, py::return_value_policy::reference)
		.def("get_qua_2_hex", &Class::getQua2Hex, py::return_value_policy::reference)
		.def("get_ver_2_hex", &Class::getVer2Hex, py::return_value_policy::reference)
		.def("get_volume", &Class::getVolume)
		.def("copy_from", &Class::copyFrom);
}

<<<<<<< HEAD
#include "Topology/JointTree.h"
template <typename TDataType>
void declare_joint_tree(py::module& m, std::string typestr) {
	using Class = dyno::JointTree<TDataType>;
	using Parent = dyno::TopologyModule;
	std::string pyclass_name = std::string("JointTree") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("copy_from", &Class::copyFrom)

		.def("scale", &Class::scale)
		.def("translate", &Class::translate)
		.def("get_global_transform", &Class::getGlobalTransform)
		.def("get_transform", &Class::getTransform)
		.def("get_quat", &Class::getQuat)
		.def("get_global_quat", &Class::getGlobalQuat)
		.def("get_coord_by_matrix", &Class::getCoordByMatrix)
		.def("get_coord_by_quat", &Class::getCoordByQuat)

		.def("get_global_coord", &Class::getGlobalCoord)

		.def("set_anim_translation", &Class::setAnimTranslation)
		.def("set_anim_rotation", &Class::setAnimRotation)
		.def("set_anim_scaling", &Class::setAnimScaling)

		.def("apply_animation_by_one", &Class::applyAnimationByOne)
		.def("apply_animation_all", &Class::applyAnimationAll)

		.def_readwrite("id", &Class::id)

		.def_readwrite("PreRotation", &Class::PreRotation)
		.def_readwrite("PreScaling", &Class::PreScaling)
		.def_readwrite("PreTranslation", &Class::PreTranslation)

		.def_readwrite("tmp", &Class::tmp)

		.def_readwrite("LclTranslation", &Class::LclTranslation)
		.def_readwrite("LclRotation", &Class::LclRotation)
		.def_readwrite("LclScaling", &Class::LclScaling)

		.def_readwrite("AnimTranslation", &Class::AnimTranslation)
		.def_readwrite("AnimRotation", &Class::AnimRotation)
		.def_readwrite("AnimScaling", &Class::AnimScaling)

		.def_readwrite("CurTranslation", &Class::CurTranslation)
		.def_readwrite("CurRotation", &Class::CurRotation)
		.def_readwrite("CurScaling", &Class::CurScaling)

		.def_readwrite("GlCoord", &Class::GlCoord)
		.def_readwrite("LastCoord", &Class::LastCoord)
		.def_readwrite("GlobalTransform", &Class::GlobalTransform)

		.def_readwrite("GlT", &Class::GlT)
		.def_readwrite("GlR", &Class::GlR)
		.def_readwrite("GlS", &Class::GlS)
		.def_readwrite("children", &Class::children)
		.def_readwrite("parent", &Class::parent);
}

#include "Topology/LevelSet.h"
template <typename TDataType>
void declare_level_set(py::module& m, std::string typestr) {
	using Class = dyno::LevelSet<TDataType>;
	using Parent = dyno::TopologyModule;
	std::string pyclass_name = std::string("SignedDistanceField") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_sdf", &Class::getSDF, py::return_value_policy::reference)
		.def("set_sdf", &Class::setSDF);
}


=======
>>>>>>> public
#include "Topology/LinearBVH.h"
template <typename TDataType>
void declare_linear_bvh(py::module& m, std::string typestr) {
	using Class = dyno::LinearBVH<TDataType>;
	std::string pyclass_name = std::string("LinearBVH") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("construct", &Class::construct)

		.def("request_intersection_number", &Class::requestIntersectionNumber)
		.def("request_intersection_ids", &Class::requestIntersectionIds)

		.def("get_root", &Class::getRoot)

		.def("get_AABB", &Class::getAABB)
		.def("get_object_idx", &Class::getObjectIdx)

		.def("get_sorted_AABBs", &Class::getSortedAABBs, py::return_value_policy::reference)
		.def("release", &Class::release);
}

#include "Topology/PolygonSet.h"
template <typename TDataType>
void declare_polygon_set(py::module& m, std::string typestr) {
	using Class = dyno::PolygonSet<TDataType>;
	using Parent = dyno::EdgeSet<TDataType>;
	std::string pyclass_name = std::string("PolygonSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_polygons", py::overload_cast<const dyno::ArrayList<dyno::uint, DeviceType::CPU>&>(&Class::setPolygons))
		.def("set_polygons", py::overload_cast<const dyno::ArrayList<dyno::uint, DeviceType::GPU>&>(&Class::setPolygons))

		.def("scale_polygon_indices", &Class::polygonIndices, py::return_value_policy::reference)
		.def("scale_vertex_2_polygon", &Class::vertex2Polygon, py::return_value_policy::reference)
		.def("scale_polygon_2_edge", &Class::polygon2Edge, py::return_value_policy::reference)
		.def("scale_edge_2_polygon", &Class::edge2Polygon, py::return_value_policy::reference)
		.def("scale_copy_from", &Class::copyFrom)
		.def("scale_is_empty", &Class::isEmpty)
		.def("scale_extract_edge_set", &Class::extractEdgeSet)
		.def("scale_extract_triangle_set", &Class::extractTriangleSet)
		.def("scale_extract_quad_set", &Class::extractQuadSet)
		.def("scale_turn_into_triangle_set", &Class::turnIntoTriangleSet)
		.def("triangle_set_to_polygon_set", &Class::triangleSetToPolygonSet);
}

#include "Topology/SimplexSet.h"
template <typename TDataType>
void declare_simplex_set(py::module& m, std::string typestr) {
	using Class = dyno::SimplexSet<TDataType>;
	using Parent = dyno::PointSet<TDataType>;
	typedef typename dyno::TopologyModule::Edge Edge;
	typedef typename dyno::TopologyModule::Triangle Triangle;
	typedef typename dyno::TopologyModule::Tetrahedron Tetrahedron;
	std::string pyclass_name = std::string("SimplexSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("copy_from", &Class::copyFrom)
		.def("is_empty", &Class::isEmpty)
		.def("set_edge_index", py::overload_cast<const dyno::Array<Edge, DeviceType::GPU>&>(&Class::setEdgeIndex))
		.def("set_edge_index", py::overload_cast<const dyno::Array<Edge, DeviceType::CPU>&>(&Class::setEdgeIndex))
		.def("set_triangle_index", py::overload_cast<const dyno::Array<Triangle, DeviceType::GPU>&>(&Class::setTriangleIndex))
		.def("set_triangle_index", py::overload_cast<const dyno::Array<Triangle, DeviceType::CPU>&>(&Class::setTriangleIndex))
		.def("set_tetrahedron_index", py::overload_cast<const dyno::Array<Tetrahedron, DeviceType::GPU>&>(&Class::setTetrahedronIndex))
		.def("set_tetrahedron_index", py::overload_cast<const dyno::Array<Tetrahedron, DeviceType::CPU>&>(&Class::setTetrahedronIndex))
		.def("extract_simplex_1d", &Class::extractSimplex1D)
		.def("extract_simplex_2d", &Class::extractSimplex2D)
		.def("extract_simplex_3d", &Class::extractSimplex3D)
		.def("extract_point_set", &Class::extractPointSet)
		.def("extract_edge_set", &Class::extractEdgeSet)
		.def("extract_triangle_set", &Class::extractTriangleSet);
}

#include "Topology/SparseGridHash.h"
template <typename TDataType>
void declare_sparse_grid_hash(py::module& m, std::string typestr) {
	using Class = dyno::SparseGridHash<TDataType>;
	std::string pyclass_name = std::string("SparseGridHash") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_space", &Class::setSpace)
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
		.def("set_space", &Class::setSpace)

		.def("construct", py::overload_cast<const dyno::Array<Coord, DeviceType::GPU>&, Real>(&Class::construct))
		.def("construct", py::overload_cast<const dyno::Array<dyno::AABB, DeviceType::GPU>&>(&Class::construct))
		.def("construct", py::overload_cast<const dyno::Array<dyno::OctreeNode, DeviceType::GPU>&>(&Class::construct))

		.def("get_level_max", &Class::getLevelMax)

		.def("query_node", &Class::queryNode)

		.def("request_level_number", &Class::requestLevelNumber)

		//.def("request_intersection_number", &Class::requestIntersectionNumber)
		//.def("reqeust_intersection_ids", &Class::reqeustIntersectionIds)

		.def("request_intersection_number_from_level", py::overload_cast<const dyno::AABB, int>(&Class::requestIntersectionNumberFromLevel))
		.def("request_intersection_number_from_level", py::overload_cast<const dyno::AABB, dyno::AABB*, int>(&Class::requestIntersectionNumberFromLevel))

		.def("request_intersection_ids_from_level", py::overload_cast<int*, const dyno::AABB, int>(&Class::reqeustIntersectionIdsFromLevel))
		.def("request_intersection_ids_from_level", py::overload_cast<int*, const dyno::AABB, dyno::AABB*, int>(&Class::reqeustIntersectionIdsFromLevel))

		.def("request_intersection_number_from_bottom", py::overload_cast<const dyno::AABB>(&Class::requestIntersectionNumberFromBottom))
		.def("request_intersection_number_from_bottom", py::overload_cast<const dyno::AABB, dyno::AABB*>(&Class::requestIntersectionNumberFromBottom))

		.def("request_intersection_ids_from_bottom", py::overload_cast<int*, const dyno::AABB>(&Class::reqeustIntersectionIdsFromBottom))
		.def("request_intersection_ids_from_bottom", py::overload_cast<int*, const dyno::AABB, dyno::AABB*>(&Class::reqeustIntersectionIdsFromBottom))

		.def("print_all_nodes", &Class::printAllNodes)
		.def("print_post_ordered_tree", &Class::printPostOrderedTree);
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
	std::string pyclass_name = std::string("TriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_triangles", py::overload_cast<std::vector<Triangle>&>(&Class::setTriangles))
		.def("set_triangles", py::overload_cast<dyno::Array<Triangle, DeviceType::GPU>&>(&Class::setTriangles))
		.def("get_triangle_2_edge", &Class::getTriangle2Edge)
		.def("get_edge_2_triangle", &Class::getEdge2Triangle)

		.def("set_normals", &Class::setNormals)
		.def("get_vertex_normals", &Class::getVertexNormals)
		.def("update_triangle_2_edge", &Class::updateTriangle2Edge)
		.def("update_edge_normal", &Class::updateEdgeNormal)
		.def("update_angle_weighted_vertex_normal", &Class::updateAngleWeightedVertexNormal)
		.def("load_obj_file", &Class::loadObjFile)
		.def("copy_from", &Class::copyFrom)
		.def("merge", &Class::merge)
		.def("is_empty", &Class::isEmpty)
		.def("clear", &Class::clear)
		.def("set_auto_update_normals", &Class::setAutoUpdateNormals)
		.def("rotate", py::overload_cast<const Coord>(&Class::rotate))
		.def("rotate", py::overload_cast<const dyno::Quat<Real>>(&Class::rotate));
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
	std::string pyclass_name = std::string("TetrahedronSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("load_tet_file", &Class::loadTetFile)
		.def("set_tetrahedrons", py::overload_cast<std::vector<Tetrahedron>&>(&Class::setTetrahedrons))
		.def("set_tetrahedrons", py::overload_cast<dyno::Array<Tetrahedron, DeviceType::GPU>&>(&Class::setTetrahedrons))
		.def("get_tetrahedrons", &Class::getTetrahedrons, py::return_value_policy::reference)
		.def("get_tri_2_tet", &Class::getTri2Tet, py::return_value_policy::reference)
		.def("get_ver_2_tet", &Class::getVer2Tet, py::return_value_policy::reference)
		.def("get_volume", &Class::getVolume)
		.def("copy_from", &Class::copyFrom)
		.def("is_empty", &Class::isEmpty);
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
		.def("copy_from", &Class::copyFrom)
		.def("get_point_neighbors", &Class::getPointNeighbors, py::return_value_policy::reference)
		.def("clear", &Class::clear);
}






void declare_texture_mesh(py::module& m);

void declare_attribute(py::module& m);

void pybind_topology(py::module& m);