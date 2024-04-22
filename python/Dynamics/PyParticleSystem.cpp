#include "PyParticleSystem.h"

void declare_func(py::module& m, std::string typestr) {
	using Class = dyno::Attribute;
}

void declare_attribute(py::module& m, std::string typestr) {
	using Class = dyno::Attribute;

	std::string pyclass_name = std::string("Attribute") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_fluid", &Class::setFluid)
		.def("set_dynamic", &Class::setDynamic);
}

#include "ParticleSystem/initializeParticleSystem.h"
void declare_paticle_system_init_static_plugin(py::module& m) {
	using Class = dyno::ParticleSystemInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("ParticleSystemInitializer");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("instance", &Class::instance);
}

void pybind_particle_system(py::module& m)
{
	declare_approximate_implicit_viscosity<dyno::DataType3f>(m, "3f");
	declare_boundary_constraint<dyno::DataType3f>(m, "3f");
	declare_implicit_viscosity<dyno::DataType3f>(m, "3f");
	declare_particle_approximation<dyno::DataType3f>(m, "3f");
	declare_iterative_densitySolver<dyno::DataType3f>(m, "3f");
	declare_linear_damping<dyno::DataType3f>(m, "3f");
	declare_particle_integrator<dyno::DataType3f>(m, "3f");
	declare_poisson_plane<dyno::DataType3f>(m, "3f");
	declare_position_based_fluid_model<dyno::DataType3f>(m, "3f");
	declare_projection_based_fluid_model<dyno::DataType3f>(m, "3f");
	declare_simple_velocity_constraint<dyno::DataType3f>(m, "3f");
	declare_summation_density<dyno::DataType3f>(m, "3f");
	declare_surface_tension<dyno::DataType3f>(m, "3f");
	declare_variational_approximate_projection<dyno::DataType3f>(m, "3f");
	declare_particle_emitter<dyno::DataType3f>(m, "3f");
	declare_particle_emitter_square<dyno::DataType3f>(m, "3f");
	declare_particle_system<dyno::DataType3f>(m, "3f");
	declare_circular_emitter<dyno::DataType3f>(m, "3f");
	declare_sampler<dyno::DataType3f>(m, "3f");
	declare_cube_sampler<dyno::DataType3f>(m, "3f");
	declare_ghost_fluid<dyno::DataType3f>(m, "3f");
	declare_ghost_particlesm<dyno::DataType3f>(m, "3f");

	declare_func(m, "");

	declare_attribute(m, "");

	declare_static_boundary<dyno::DataType3f>(m, "3f");






	declare_particle_fluid<dyno::DataType3f>(m, "3f");
	declare_particle_elastic_body<dyno::DataType3f>(m, "3f");
	declare_make_particle_system<dyno::DataType3f>(m, "3f");




}