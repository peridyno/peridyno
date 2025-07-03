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
		.def("getNodeType", &Class::getNodeType)
		.def("varOutputPath", &Class::varOutputPath, py::return_value_policy::reference)
		.def("varStartFrame", &Class::varStartFrame, py::return_value_policy::reference)
		.def("varEndFrame", &Class::varEndFrame, py::return_value_policy::reference)
		.def("varFrameStep", &Class::varFrameStep, py::return_value_policy::reference)
		.def("inPolygonSet", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference);

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
		.def("getNodeType", &Class::getNodeType)
		.def("varFileName", &Class::varFileName, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference)
		.def("varSequence", &Class::varSequence, py::return_value_policy::reference)
		.def("varVelocity", &Class::varVelocity, py::return_value_policy::reference)
		.def("varCenter", &Class::varCenter, py::return_value_policy::reference)
		.def("varAngularVelocity", &Class::varAngularVelocity, py::return_value_policy::reference)
		.def("stateTopology", &Class::stateTopology, py::return_value_policy::reference);
}

#include "ObjIO/ObjPointLoader.h"
template <typename TDataType>
void declare_Obj_point(py::module& m, std::string typestr) {
	using Class = dyno::ObjPoint<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ObjPoint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("varLocation", &Class::varLocation, py::return_value_policy::reference)
		.def("varRotation", &Class::varRotation, py::return_value_policy::reference)
		.def("varScale", &Class::varScale, py::return_value_policy::reference)

		.def("varFileName", &Class::varFileName, py::return_value_policy::reference)

		.def("varRadius", &Class::varRadius, py::return_value_policy::reference)
		.def("outPointSet", &Class::outPointSet, py::return_value_policy::reference)
		.def("varSequence", &Class::varSequence, py::return_value_policy::reference)
		.def("varVelocity", &Class::varVelocity, py::return_value_policy::reference)
		.def("varAngularVelocity", &Class::varAngularVelocity, py::return_value_policy::reference)
		.def("stateTopology", &Class::stateTopology, py::return_value_policy::reference);
}

#include "ObjIO/PLYexporter.h"
template <typename TDataType>
void declare_PlyExporter(py::module& m, std::string typestr) {
	using Class = dyno::PlyExporter<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("PlyExporter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("varOutputPath", &Class::varOutputPath, py::return_value_policy::reference)
		.def("varFrameStep", &Class::varFrameStep, py::return_value_policy::reference)
		.def("varReCount", &Class::varReCount, py::return_value_policy::reference)
		.def("inTopology", &Class::inTopology, py::return_value_policy::reference)
		.def("inVec3f", &Class::inVec3f, py::return_value_policy::reference)
		.def("inMatrix1", &Class::inMatrix1, py::return_value_policy::reference)
		.def("inMatrix2", &Class::inMatrix2, py::return_value_policy::reference);
}

void pybind_objIO(py::module& m);