#include "PyMultiphysics.h"

#include "Multiphysics/initializeMultiphysics.h"
void declare_multiphysics_initializer(py::module& m) {
	using Class = dyno::MultiphysicsInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("MultiphysicsInitializer");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("instance", &Class::instance);
}

void pybind_multiphysics(py::module& m)
{
	declare_adaptive_boundary<dyno::DataType3f>(m, "3f");
	declare_compute_surface_level_set<dyno::DataType3f>(m, "3f");
	declare_particle_skinning<dyno::DataType3f>(m, "3f");
	declare_volume_boundary<dyno::DataType3f>(m, "3f");
	declare_multiphysics_initializer(m);
}