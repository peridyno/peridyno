#include "PyModeling.h"

#include "initializeModeling.h"
void declare_modeling_initializer(py::module& m) {
	using Class = dyno::ModelingInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("ModelingInitializer");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("instance", &Class::instance);
}

void pybind_modeling(py::module& m) {
	//declare_var<dyno::TOrientedBox3D<Real>>(m, "TOrientedBox3D");
	declare_cube_model<dyno::DataType3f>(m, "3f");
	declare_plane_model<dyno::DataType3f>(m, "3f");
	declare_sphere_model<dyno::DataType3f>(m, "3f");
	declare_static_triangular_mesh<dyno::DataType3f>(m, "3f");
	declare_modeling_initializer(m);
}