#include "PyParticleSystem.h"

#include "ParticleSystem/initializeParticleSystem.h"
void declare_particle_system_initializer(py::module& m) {
	using Class = dyno::ParticleSystemInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("ParticleSystemInitializer");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("instance", &Class::instance);
}

void pybind_particle_system(py::module& m)
{
	// Emitters
	declare_particle_emitter<dyno::DataType3f>(m, "3f");
	declare_circular_emitter<dyno::DataType3f>(m, "3f");
	declare_poisson_emitter<dyno::DataType3f>(m, "3f");
	declare_square_emitter<dyno::DataType3f>(m, "3f");

	//Module
	declare_approximate_implicit_viscosity<dyno::DataType3f>(m, "3f");
	declare_boundary_constraint<dyno::DataType3f>(m, "3f");
	declare_particle_approximation<dyno::DataType3f>(m, "3f");
	declare_divergence_free_sph_solver<dyno::DataType3f>(m, "3f");
	declare_implicit_ISPH<dyno::DataType3f>(m, "3f");
	declare_implicit_viscosity<dyno::DataType3f>(m, "3f");
	declare_iterative_densitySolver<dyno::DataType3f>(m, "3f");
	declare_linear_damping<dyno::DataType3f>(m, "3f");
	declare_normal_force<dyno::DataType3f>(m, "3f");
	declare_particle_integrator<dyno::DataType3f>(m, "3f");
	declare_position_based_fluid_model<dyno::DataType3f>(m, "3f");
	declare_projection_based_fluid_model<dyno::DataType3f>(m, "3f");
	declare_simple_velocity_constraint<dyno::DataType3f>(m, "3f");
	declare_summation_density<dyno::DataType3f>(m, "3f");
	declare_surface_tension<dyno::DataType3f>(m, "3f");
	declare_variational_approximate_projection<dyno::DataType3f>(m, "3f");

	//Particle System
	declare_particle_system<dyno::DataType3f>(m, "3f");
	declare_particle_fluid<dyno::DataType3f>(m, "3f");
	declare_ghost_fluid<dyno::DataType3f>(m, "3f");
	declare_ghost_particles<dyno::DataType3f>(m, "3f");
	declare_make_ghost_particles<dyno::DataType3f>(m, "3f");
	declare_make_particle_system<dyno::DataType3f>(m, "3f");
	declare_particle_system_helper<dyno::DataType3f>(m, "3f");
	declare_sdf_sampler<dyno::DataType3f>(m, "3f");
	declare_poisson_disk_sampling<dyno::DataType3f>(m, "3f");

	declare_particle_system_initializer(m);
}