#include "PyIO.h"

#include "Gmsh_IO/gmsh.h"
void declare_gmsh(py::module& m) {
	using Class = dyno::Gmsh;
	std::string pyclass_name = std::string("Gmsh");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("load_file", &Class::loadFile);
}

#include "Smesh_IO/smesh.h"
void declare_smesh(py::module& m) {
	using Class = dyno::Smesh;
	std::string pyclass_name = std::string("Smesh");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("load_file", &Class::loadFile)
		.def("load_node_file", &Class::loadNodeFile)
		.def("load_edge_file", &Class::loadEdgeFile)
		.def("load_triangle_file", &Class::loadTriangleFile)
		.def("load_tet_file", &Class::loadTetFile);
}

#include "initializeIO.h"
void declare_io_initializer(py::module& m) {
	using Class = dyno::IOInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("IOInitializer");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("instance", &Class::instance);
}

void pybind_io(py::module& m)
{
	declare_gmsh(m);
	declare_smesh(m);
	declare_io_initializer(m);
	declare_eigen_value_writer<dyno::DataType3f>(m, "3f");
	declare_geometry_loader<dyno::DataType3f>(m, "3f");
	declare_particle_writer<dyno::DataType3f>(m, "3f");
	declare_points_loader<dyno::DataType3f>(m, "3f");
	declare_surface_mesh_loader<dyno::DataType3f>(m, "3f");
	//declare_tetra_mesh_writer<dyno::DataType3f>(m, "3f");
	declare_tetra_mesh_writer_fracture<dyno::DataType3f>(m, "3f");
	declare_triangle_mesh_writer<dyno::DataType3f>(m, "3f");
}