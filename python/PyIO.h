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
		.def("inTransform", &Class::inTransform, py::return_value_policy::reference);
}

#include "GeometryLoader.h"
template <typename TDataType>
void declare_geometry_loader(py::module& m, std::string typestr) {
	using Class = dyno::GeometryLoader<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("GeometryLoader") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varFileName", &Class::varFileName, py::return_value_policy::reference);
}

#include "ParticleWriter.h"
template <typename TDataType>
void declare_particle_writer(py::module& m, std::string typestr) {
	using Class = dyno::ParticleWriter<TDataType>;
	using Parent = dyno::OutputModule;
	std::string pyclass_name = std::string("ParticleWriter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>PW(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	PW.def(py::init<>())
		.def("OutputASCII", &Class::OutputASCII)
		.def("OutputBinary", &Class::OutputBinary)
		.def("output", &Class::output)
		.def("inPointSet", &Class::inPointSet, py::return_value_policy::reference)
		.def("varFileType", &Class::varFileType, py::return_value_policy::reference);

	py::enum_<typename Class::OpenType>(PW, "OpenType")
		.value("ASCII", Class::OpenType::ASCII)
		.value("binary", Class::OpenType::binary)
		.export_values();
}

#include "PointsLoader.h"
template <typename TDataType>
void declare_points_loader(py::module& m, std::string typestr) {
	using Class = dyno::PointsLoader<TDataType>;
	using Parent = dyno::GeometryLoader<TDataType>;

	class PointsLoaderTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PointsLoader<TDataType>,
				resetStates
			);
		}
	};

	class PointsLoaderPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("PointsLoader") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("outPointSet", &Class::outPointSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &PointsLoaderPublicist::resetStates);
}

#include "StaticMeshLoader.h"
template <typename TDataType>
void declare_static_mesh_loader(py::module& m, std::string typestr) {
	using Class = dyno::StaticMeshLoader<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("StaticMeshLoader") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("varFileName", &Class::varFileName, py::return_value_policy::reference)
		.def("stateInitialTriangleSet", &Class::stateInitialTriangleSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference);
}

#include "TetraMeshWriter.h"
template <typename TDataType>
void declare_tetra_mesh_writer(py::module& m, std::string typestr) {
	using Class = dyno::TetraMeshWriter<TDataType>;
	using Parent = dyno::OutputModule;

	class TetraMeshWriterTrampoline : public Class
	{
	public:
		using Class::Class;

		void output() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TetraMeshWriter<TDataType>,
				output
			);
		}
	};

	class TetraMeshWriterPublicist : public Class
	{
	public:
		using Class::output;
	};

	std::string pyclass_name = std::string("TetraMeshWriter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTetrahedronSet", &Class::inTetrahedronSet, py::return_value_policy::reference)
		// protected
		.def("output", &TetraMeshWriterPublicist::output);
}

#include "TextureMeshLoader.h"
template <typename TDataType>
void declare_texture_mesh_loader(py::module& m, std::string typestr) {
	using Class = dyno::TextureMeshLoader;
	using Parent = dyno::ParametricModel<TDataType>;

	class TextureMeshLoaderTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TextureMeshLoader,
				resetStates
			);
		}
	};

	class TextureMeshLoaderPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("TextureMeshLoader") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("varFileName", &Class::varFileName, py::return_value_policy::reference)
		.def("stateTextureMesh", &Class::stateTextureMesh, py::return_value_policy::reference)
		// protected
		.def("resetStates", &TextureMeshLoaderPublicist::resetStates);
}

#include "TriangleMeshWriter.h"
template <typename TDataType>
void declare_triangle_mesh_writer(py::module& m, std::string typestr) {
	using Class = dyno::TriangleMeshWriter<TDataType>;
	using Parent = dyno::OutputModule;
	std::string pyclass_name = std::string("TriangleMeshWriter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>TMW(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	TMW.def(py::init<>())
		.def("outputSurfaceMesh", &Class::outputSurfaceMesh)
		.def("outputPointCloud", &Class::outputPointCloud)
		.def("output", &Class::output)
		.def("inTopology", &Class::inTopology, py::return_value_policy::reference)
		.def("varOutputType", &Class::varOutputType, py::return_value_policy::reference);

	py::enum_<typename Class::OutputType>(TMW, "OutputType")
		.value("TriangleMesh", Class::OutputType::TriangleMesh)
		.value("PointCloud", Class::OutputType::PointCloud)
		.export_values();
}

void declare_gmsh(py::module& m);

void declare_smesh(py::module& m);

void declare_image_loader(py::module& m);

void pybind_io(py::module& m);