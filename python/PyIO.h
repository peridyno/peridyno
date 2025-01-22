#pragma once
#include "PyCommon.h"

#include "EigenValueWriter.h"
template <typename TDataType>
void declare_eigen_value_writer(py::module& m, std::string typestr) {
	using Class = dyno::EigenValueWriter<TDataType>;
	using Parent = dyno::OutputModule;
	std::string pyclass_name = std::string("EigenValueWriter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("output", &Class::output)
		.def("in_transform", &Class::inTransform, py::return_value_policy::reference);
}

#include "GeometryLoader.h"
template <typename TDataType>
void declare_geometry_loader(py::module& m, std::string typestr) {
	using Class = dyno::GeometryLoader<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("GeometryLoader") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_file_name", &Class::varFileName, py::return_value_policy::reference);
}

#include "ParticleWriter.h"
template <typename TDataType>
void declare_particle_writer(py::module& m, std::string typestr) {
	using Class = dyno::ParticleWriter<TDataType>;
	using Parent = dyno::OutputModule;
	std::string pyclass_name = std::string("ParticleWriter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>PW(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	PW.def(py::init<>())
		.def("output_ascii", &Class::OutputASCII)
		.def("output_binary", &Class::OutputBinary)
		.def("output", &Class::output)
		.def("in_point_set", &Class::inPointSet, py::return_value_policy::reference)
		.def("var_file_type", &Class::varFileType, py::return_value_policy::reference);

	py::enum_<typename Class::OpenType>(PW, "OpenType")
		.value("ASCII", Class::OpenType::ASCII)
		.value("binary", Class::OpenType::binary);
}

#include "PointsLoader.h"
template <typename TDataType>
void declare_points_loader(py::module& m, std::string typestr) {
	using Class = dyno::PointsLoader<TDataType>;
	using Parent = dyno::GeometryLoader<TDataType>;
	std::string pyclass_name = std::string("PointsLoader") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("out_point_set", &Class::outPointSet, py::return_value_policy::reference);
}

#include "TetraMeshWriter.h"
template <typename TDataType>
void declare_tetra_mesh_writer(py::module& m, std::string typestr) {
	using Class = dyno::TetraMeshWriter<TDataType>;
	using Parent = dyno::OutputModule;
	std::string pyclass_name = std::string("TetraMeshWriter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_name_prefix", &Class::setNamePrefix)
		.def("set_output_path", &Class::setOutputPath)
		.def("set_tetrahedron_set_ptr", &Class::setTetrahedronSetPtr)
		.def("update_ptr", &Class::updatePtr)
		.def("output_surface_mesh", &Class::outputSurfaceMesh)
		.def("output", &Class::output);
}

#include "TetraMeshWriterFracture.h"
template <typename TDataType>
void declare_tetra_mesh_writer_fracture(py::module& m, std::string typestr) {
	using Class = dyno::TetraMeshWriterFracture<TDataType>;
	using Parent = dyno::OutputModule;
	std::string pyclass_name = std::string("TetraMeshWriterFracture") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("load_uvs", &Class::loadUVs)
		.def("set_tetrahedron_set_ptr", &Class::setTetrahedronSetPtr)
		.def("update_ptr", &Class::updatePtr)
		.def("output_surface_mesh", &Class::outputSurfaceMesh);
}

#include "TriangleMeshWriter.h"
template <typename TDataType>
void declare_triangle_mesh_writer(py::module& m, std::string typestr) {
	using Class = dyno::TriangleMeshWriter<TDataType>;
	using Parent = dyno::OutputModule;
	std::string pyclass_name = std::string("TriangleMeshWriter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>TMW(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	TMW.def(py::init<>())
		.def("output_surface_mesh", &Class::outputSurfaceMesh)
		.def("output_point_cloud", &Class::outputPointCloud)
		.def("output", &Class::output)
		.def("in_topology", &Class::inTopology, py::return_value_policy::reference)
		.def("var_output_type", &Class::varOutputType, py::return_value_policy::reference);

	py::enum_<typename Class::OutputType>(TMW, "OutputType")
		.value("TriangleMesh", Class::OutputType::TriangleMesh)
		.value("PointCloud", Class::OutputType::PointCloud);
}

void declare_gmsh(py::module& m);

void declare_smesh(py::module& m);

void declare_io_initializer(py::module& m);

void pybind_io(py::module& m);