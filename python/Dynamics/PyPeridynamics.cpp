#include "PyPeridynamics.h"

void pybind_peridynamics(py::module& m)
{
	//Module
	declare_calculate_normal_sdf<dyno::DataType3f>(m, "3f");
	declare_contact_rule<dyno::DataType3f>(m, "3f");
	declare_linear_elasticity_solver<dyno::DataType3f>(m, "3f");
	declare_co_semi_implicit_hyperelasticity_solver<dyno::DataType3f>(m, "3f");
	declare_dampling_particle_integrator<dyno::DataType3f>(m, "3f");
	declare_drag_surface_interaction<dyno::DataType3f>(m, "3f");
	declare_drag_vertex_interaction<dyno::DataType3f>(m, "3f");
	declare_elastoplasticity_module<dyno::DataType3f>(m, "3f");
	declare_fixed_points<dyno::DataType3f>(m, "3f");
	declare_fracture_module<dyno::DataType3f>(m, "3f");
	declare_granular_module<dyno::DataType3f>(m, "3f");
	//declare_one_dim_elasticity_module<dyno::DataType3f>(m, "3f");
	declare_peridynamics<dyno::DataType3f>(m, "3f");
	declare_semi_implicit_hyperelasticity_solver<dyno::DataType3f>(m, "3f");

	declare_triangular_system<dyno::DataType3f>(m, "3f");
	declare_codimensionalPD<dyno::DataType3f>(m, "3f");

	declare_bond<dyno::DataType3f>(m, "3f");
	declare_cloth<dyno::DataType3f>(m, "3f");
	declare_elastic_body<dyno::DataType3f>(m, "3f");
	declare_elastoplastic_body<dyno::DataType3f>(m, "3f");
	declare_tetrahedral_system<dyno::DataType3f>(m, "3f");
	declare_hyperelastic_body<dyno::DataType3f>(m, "3f");
	//declare_peridynamics_initializer<dyno::DataType3f>(m, "3f");
	declare_thread_system<dyno::DataType3f>(m, "3f");
	declare_thread<dyno::DataType3f>(m, "3f");
}