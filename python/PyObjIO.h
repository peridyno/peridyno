/*
 * @Author: unibeam98
 * @Date: 2024-07-08 17:57:21
 * @LastEditors: unibeam98
 * @LastEditTime: 2025-01-16 19:42:13
 * @FilePath: \peridyno-web\python\PyObjIO.h
 */
#pragma once
#include "PyCommon.h"

#include "ObjIO/OBJexporter.h"
template <typename TDataType>
void declare_Obj_exporter(py::module& m, std::string typestr) {
	using Class = dyno::ObjExporter<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ObjExporter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>OBJE(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	OBJE.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)
		.def("var_output_path", &Class::varOutputPath, py::return_value_policy::reference)
		.def("var_start_frame", &Class::varStartFrame, py::return_value_policy::reference)
		.def("var_end_frame", &Class::varEndFrame, py::return_value_policy::reference)
		.def("var_frame_step", &Class::varFrameStep, py::return_value_policy::reference)
		.def("in_polygon_set", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference);

	py::enum_<typename Class::OutputType>(OBJE, "OutputType")
		.value("Mesh", Class::OutputType::Mesh)
		.value("PointCloud", Class::OutputType::PointCloud)
		.export_values();
}

#include "ObjIO/ObjLoader.h"
template <typename TDataType>
void declare_Obj_loader(py::module& m, std::string typestr) {
	using Class = dyno::ObjLoader<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("ObjLoader") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)
		.def("var_file_name", &Class::varFileName, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference)
		.def("var_sequence", &Class::varSequence, py::return_value_policy::reference)
		.def("var_velocity", &Class::varVelocity, py::return_value_policy::reference)
		.def("var_center", &Class::varCenter, py::return_value_policy::reference)
		.def("var_angular_velocity", &Class::varAngularVelocity, py::return_value_policy::reference)
		.def("state_topology", &Class::stateTopology, py::return_value_policy::reference);
}

#include "ObjIO/ObjPointLoader.h"
template <typename TDataType>
void declare_Obj_point(py::module& m, std::string typestr) {
	using Class = dyno::ObjPoint<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ObjPoint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)
		.def("var_location", &Class::varLocation, py::return_value_policy::reference)
		.def("var_rotation", &Class::varRotation, py::return_value_policy::reference)
		.def("var_scale", &Class::varScale, py::return_value_policy::reference)

		.def("var_file_name", &Class::varFileName, py::return_value_policy::reference)

		.def("var_radius", &Class::varRadius, py::return_value_policy::reference)
		.def("out_point_set", &Class::outPointSet, py::return_value_policy::reference)
		.def("var_sequence", &Class::varSequence, py::return_value_policy::reference)
		.def("var_velocity", &Class::varVelocity, py::return_value_policy::reference)
		.def("var_angular_velocity", &Class::varAngularVelocity, py::return_value_policy::reference)
		.def("state_topology", &Class::stateTopology, py::return_value_policy::reference);
}

#include "ObjIO/PLYexporter.h"
template <typename TDataType>
void declare_PlyExporter(py::module& m, std::string typestr) {
	using Class = dyno::PlyExporter<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("PlyExporter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)
		.def("var_output_path", &Class::varOutputPath, py::return_value_policy::reference)
		.def("var_frame_step", &Class::varFrameStep, py::return_value_policy::reference)
		.def("var_re_count", &Class::varReCount, py::return_value_policy::reference)
		.def("in_topology", &Class::inTopology, py::return_value_policy::reference)
		.def("in_vec3f", &Class::inVec3f, py::return_value_policy::reference)
		.def("in_matrix1", &Class::inMatrix1, py::return_value_policy::reference)
		.def("in_matrix2", &Class::inMatrix2, py::return_value_policy::reference);
}

void pybind_objIO(py::module& m);