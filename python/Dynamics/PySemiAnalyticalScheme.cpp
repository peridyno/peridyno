#include "PySemiAnalyticalScheme.h"

#include "SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h"
void declare_semi_analytical_scheme_initializer(py::module& m) {
	using Class = dyno::SemiAnalyticalSchemeInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("SemiAnalyticalSchemeInitializer");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("instance", &Class::instance);
}

void pybind_semi_analytical_scheme(py::module& m)
{
	//declare_semi_analytical_sfi_node<dyno::DataType3f>(m, "3f");
	declare_compute_particle_anisotropy<dyno::DataType3f>(m, "3f");
	declare_semi_analytical_scheme_initializer(m);
	declare_semi_analytical_incompressibility_module<dyno::DataType3f>(m, "3f");
	declare_semi_analytical_incompressible_fluid_model<dyno::DataType3f>(m, "3f");
	declare_semi_analytical_particle_shifting<dyno::DataType3f>(m, "3f");
	declare_semi_analytical_pbd<dyno::DataType3f>(m, "3f");
	declare_semi_analytical_position_based_fluid_model<dyno::DataType3f>(m, "3f");
	declare_semi_analytical_sfi_node<dyno::DataType3f>(m, "3f");
	declare_semi_analytical_summation_density<dyno::DataType3f>(m, "3f");
	declare_semi_analytical_surface_tension_model<dyno::DataType3f>(m, "3f");
	declare_triangular_mesh_boundary<dyno::DataType3f>(m, "3f");
	declare_triangular_mesh_constraint<dyno::DataType3f>(m, "3f");


}