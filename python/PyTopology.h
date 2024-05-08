#pragma once
#include "PyCommon.h"

#include "Topology/PointSet.h"
template <typename TDataType>
void declare_pointset(py::module& m, std::string typestr) {
	using Class = dyno::PointSet<TDataType>;
	using Parent = dyno::TopologyModule;
	std::string pyclass_name = std::string("PointSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("copy_from", &Class::copyFrom)
		//.def("set_points", &Class::setPoints)
		.def("set_size", &Class::setSize)
		.def("get_point_size", &Class::getPointSize)
		.def("request_bounding_box", &Class::requestBoundingBox)
		//.def("scale",&Class::scale)
		.def("translate", &Class::translate)
		//.def("rotate", &Class::rotate)
		.def("load_obj_file", &Class::loadObjFile)
		.def("is_empty", &Class::isEmpty)
		.def("clear", &Class::clear)
		.def("get_points", &Class::getPoints);
}

#include "Topology/EdgeSet.h"
template <typename TDataType>
void declare_edgeSet(py::module& m, std::string typestr) {
	using Class = dyno::EdgeSet<TDataType>;
	using Parent = dyno::PointSet<TDataType>;
	std::string pyclass_name = std::string("EdgeSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//.def("set_edges", &Class::setEdges);
		.def("request_point_neighbors", &Class::requestPointNeighbors)
		.def("get_edges", &Class::getEdges)
		.def("vertex_2_edge", &Class::vertex2Edge)
		.def("copy_from", &Class::copyFrom)
		.def("is_empty", &Class::isEmpty)
		.def("clear", &Class::clear)
		.def("load_smesh_file", &Class::loadSmeshFile);

}

#include "Topology/TriangleSet.h"
template <typename TDataType>
void declare_triangleSet(py::module& m, std::string typestr) {
	using Class = dyno::TriangleSet<TDataType>;
	using Parent = dyno::EdgeSet<TDataType>;
	std::string pyclass_name = std::string("TriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//.def("set_triangles", &Class::setTriangles);
		.def("get_triangles", &Class::getTriangles)
		//.def("get_vertex_2_triangles", &Class::getVertex2Triangles)
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
		.def("set_auto_update_normals", &Class::setAutoUpdateNormals);
	//.def("rotate", &Class::rotate)
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

#include "Module/CalculateNorm.h"
template <typename TDataType>
void declare_calculate_norm(py::module& m, std::string typestr) {
	using Class = dyno::CalculateNorm<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("CalculateNorm") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_vec", &Class::inVec, py::return_value_policy::reference)
		.def("out_norm", &Class::outNorm, py::return_value_policy::reference);
}

#include "Mapping/HeightFieldToTriangleSet.h"
template <typename TDataType>
void declare_height_field_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::HeightFieldToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("HeightFieldToTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_scale", &Class::varScale, py::return_value_policy::reference)
		.def("var_translation", &Class::varTranslation, py::return_value_policy::reference)
		.def("in_height_field", &Class::inHeightField, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference);
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

#include "Mapping/ContactsToEdgeSet.h"
template<typename TDataType>
void declare_contacts_to_edge_set(py::module& m, std::string typestr) {
	using Class = dyno::ContactsToEdgeSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("ContactsToEdgeSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_contacts", &Class::inContacts, py::return_value_policy::reference)
		.def("var_scale", &Class::varScale, py::return_value_policy::reference)
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
		.value("OCTREE", Class::Spatial::OCTREE);
}




void pybind_topology(py::module& m);