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

void pybind_particle_system(py::module& m)
{
	declare_func(m, "");

	declare_attribute(m, "");

	declare_multi_node_port<dyno::ParticleEmitter<dyno::DataType3f>>(m, "ParticleEmitter3f");
	declare_multi_node_port<dyno::ParticleSystem<dyno::DataType3f>>(m, "ParticleSystem3f");
	declare_multi_node_port<dyno::TriangularSystem<dyno::DataType3f>>(m, "TriangularSystem3f");

	declare_static_boundary<dyno::DataType3f>(m, "3f");
	declare_sampler<dyno::DataType3f>(m, "3f");
	declare_cube_sampler<dyno::DataType3f>(m, "3f");

	declare_particle_system<dyno::DataType3f>(m, "3f");
	declare_particle_emitter<dyno::DataType3f>(m, "3f");
	declare_particle_emitter_square<dyno::DataType3f>(m, "3f");
	declare_particle_fluid<dyno::DataType3f>(m, "3f");
	declare_particle_elastic_body<dyno::DataType3f>(m, "3f");
	declare_make_particle_system<dyno::DataType3f>(m, "3f");

	declare_ghost_particlesm<dyno::DataType3f>(m, "3f");

	declare_circular_emitter<dyno::DataType3f>(m, "3f");
}