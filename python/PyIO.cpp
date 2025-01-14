#include "PyIO.h"

#include "Gmsh_IO/gmsh.h"
void declare_gmsh(py::module& m) {
	using Class = dyno::Gmsh;
	std::string pyclass_name = std::string("Gmsh");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("load_file", &Class::loadFile)
		.def_readwrite("m_points", &Class::m_points)
		.def_readwrite("m_tets", &Class::m_tets);
}

#include "Smesh_IO/smesh.h"
void declare_smesh(py::module& m) {
	using Class = dyno::Smesh;
	std::string pyclass_name = std::string("Smesh");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("load_file", &Class::loadFile)
		.def("load_node_file", &Class::loadNodeFile)
		.def("load_edge_file", &Class::loadEdgeFile)
		.def("load_triangle_file", &Class::loadTriangleFile)
		.def("load_tet_file", &Class::loadTetFile)
		.def_readwrite("m_points", &Class::m_points)
		.def_readwrite("m_edges", &Class::m_edges)
		.def_readwrite("m_triangles", &Class::m_triangles)
		.def_readwrite("m_quads", &Class::m_quads)
		.def_readwrite("m_tets", &Class::m_tets)
		.def_readwrite("m_hexs", &Class::m_hexs);
}

#include "ImageLoader.h"
void declare_image_loader(py::module& m)
{
	using Class = dyno::ImageLoader;
	std::string pyclass_name = std::string("ImageLoader");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("load_image", &Class::loadImage);
}

void pybind_io(py::module& m)
{
	declare_gmsh(m);
	declare_smesh(m);
	declare_image_loader(m);

	declare_eigen_value_writer<dyno::DataType3f>(m, "3f");
	declare_geometry_loader<dyno::DataType3f>(m, "3f");
	declare_particle_writer<dyno::DataType3f>(m, "3f");
	declare_points_loader<dyno::DataType3f>(m, "3f");
	declare_surface_mesh_loader<dyno::DataType3f>(m, "3f");
	declare_tetra_mesh_writer<dyno::DataType3f>(m, "3f");
	declare_tetra_mesh_writer_fracture<dyno::DataType3f>(m, "3f");
	declare_triangle_mesh_writer<dyno::DataType3f>(m, "3f");
}