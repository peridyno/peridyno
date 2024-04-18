#include "PyPeridynamics.h"

void pybind_peridynamics(py::module& m)
{
	declare_triangular_system<dyno::DataType3f>(m, "3f");
	declare_codimensionalPD<dyno::DataType3f>(m, "3f");
	declare_calculate_normal_sdf<dyno::DataType3f>(m, "3f");
	declare_contact_rule<dyno::DataType3f>(m, "3f");
	declare_linear_elasticity_solver<dyno::DataType3f>(m, "3f");
	declare_co_semi_implicit_hyperelasticity_solver<dyno::DataType3f>(m, "3f");
	declare_dampling_particle_integrator<dyno::DataType3f>(m, "3f");
	declare_drag_surface_interaction<dyno::DataType3f>(m, "3f");
}