#include "PyDualParticleSystem.h"

#include "DualParticleSystem/initializeDualParticleSystem.h"
void declare_dual_particle_system_initializer(py::module& m) {
	using Class = dyno::DualParticleSystemInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("DualParticleSystemInitializer");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("instance", &Class::instance);
}

void pybind_dual_particle_system(py::module& m)
{
	declare_dual_particle_fluid_system<dyno::DataType3f>(m, "3f");
	declare_dual_particle_isph_module<dyno::DataType3f>(m, "3f");
	declare_energy_analyish<dyno::DataType3f>(m, "3f");
	declare_paticle_uniform_analysis<dyno::DataType3f>(m, "3f");
	declare_virtual_particle_generator<dyno::DataType3f>(m, "3f");
	declare_virtual_colocation_strategy<dyno::DataType3f>(m, "3f");
	declare_virtual_particle_shifting_strategy<dyno::DataType3f>(m, "3f");
	declare_virtual_spatially_adaptive_strategy<dyno::DataType3f>(m, "3f");
}