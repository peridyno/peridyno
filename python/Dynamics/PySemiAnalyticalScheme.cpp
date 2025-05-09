#include "PySemiAnalyticalScheme.h"

void pybind_semi_analytical_scheme(py::module& m)
{
	//declare_semi_analytical_sfi_node<dyno::DataType3f>(m, "3f");
	declare_compute_particle_anisotropy<dyno::DataType3f>(m, "3f");
	declare_particle_relaxtion_on_mesh<dyno::DataType3f>(m, "3f");
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